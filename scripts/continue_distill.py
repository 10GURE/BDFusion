#!/usr/bin/env python3
"""手动蒸馏续训：batch_size=1，teacher 推理后立刻 del 中间结果防 OOM。"""
import os, torch, time
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

STUDENT_CONFIG = "./informationSynthesis/Image2Image_Lite.yaml"
TEACHER_CKPT   = "./checkpoint/last.ckpt"
RESUME_FROM    = "./checkpoint/distill/manual-epoch=32-val=0.1057.ckpt"
OUTPUT_DIR     = "./checkpoint/distill"
EXTRA_EPOCHS   = 60
LR             = 3.0e-7
BATCH_SIZE     = 1
SAVE_EVERY     = 5

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"续训蒸馏: {EXTRA_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")

    # -- Teacher (frozen, on GPU) --
    tc = OmegaConf.load("./informationSynthesis/Image2Image.yaml")
    teacher = instantiate_from_config(tc.model)
    ts = torch.load(TEACHER_CKPT, map_location='cpu')['state_dict']
    remap = {}
    for k,v in list(ts.items()):
        if '.to_k.' in k:
            nk=k.replace('.to_k.','.to_v.')
            if nk not in ts: remap[nk]=v.clone()
    ts.update(remap)
    teacher.load_state_dict(ts, strict=False)
    for p in teacher.parameters(): p.requires_grad = False
    teacher = teacher.cuda().eval()

    # -- Student (trainable) --
    sc = OmegaConf.load(STUDENT_CONFIG)
    student = instantiate_from_config(sc.model)
    ckpt = torch.load(RESUME_FROM, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd = {k[8:] if k.startswith('student.') else k: v for k,v in sd.items()}
    for k in list(ts.keys()):
        if k.startswith('first_stage_model.') and k not in sd: sd[k] = ts[k]
    student.load_state_dict(sd, strict=False)
    for n,p in student.named_parameters():
        if n.startswith('first_stage_model.') or n.startswith('cond_stage_model.'):
            p.requires_grad = False
    student = student.cuda().train()

    opt = torch.optim.AdamW(student.model.parameters(), lr=LR)
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.0f}M, Student trainable: {sum(p.numel() for p in student.parameters() if p.requires_grad)/1e6:.0f}M")

    # -- Data --
    dc = OmegaConf.load(STUDENT_CONFIG).data
    train_ds = instantiate_from_config(dc.params.train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_ds = instantiate_from_config(dc.params.validation)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_val = float('inf')
    T = student.num_timesteps

    for epoch in range(EXTRA_EPOCHS):
        student.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            inputs = student.get_input(batch, student.first_stage_key)
            if len(inputs) == 3: x, c1, c2 = inputs
            elif len(inputs) == 4: x, c1, c2, _ = inputs
            else: continue

            t = torch.randint(0, T, (x.shape[0],), device='cuda').long()
            noise = torch.randn_like(x)
            x_noisy = student.q_sample(x_start=x, t=t, noise=noise)

            s_out = student.apply_model(x_noisy, t, c1, c2)

            with torch.no_grad():
                t_out = teacher.apply_model(x_noisy, t, c1, c2)

            loss = torch.nn.functional.mse_loss(s_out, noise) + 0.3 * torch.nn.functional.mse_loss(s_out, t_out)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            del s_out, t_out, x_noisy, t, noise, loss

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = student.get_input(batch, student.first_stage_key)
                if len(inputs) == 3: x, c1, c2 = inputs
                elif len(inputs) == 4: x, c1, c2, _ = inputs
                else: continue
                t = torch.randint(0, T, (x.shape[0],), device='cuda').long()
                noise = torch.randn_like(x)
                x_noisy = student.q_sample(x_start=x, t=t, noise=noise)
                s_out = student.apply_model(x_noisy, t, c1, c2)
                val_loss += torch.nn.functional.mse_loss(s_out, noise).item()
                del s_out, x_noisy, t, noise
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{EXTRA_EPOCHS} | loss={avg_loss:.4f} val={val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict': student.state_dict()},
                       os.path.join(OUTPUT_DIR, f"manual-epoch={epoch+1:02d}-val={val_loss:.4f}.ckpt"))
            print(f"  → saved best")

        if (epoch+1) % SAVE_EVERY == 0:
            torch.save({'state_dict': student.state_dict()},
                       os.path.join(OUTPUT_DIR, f"manual-epoch={epoch+1:02d}.ckpt"))

    print(f"Done. Best val={best_val:.4f}")

if __name__ == "__main__":
    main()
