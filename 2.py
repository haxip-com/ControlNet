def process(self, detected_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    print(num_samples, type(num_samples))
    print("111")
    with torch.no_grad():
        detected_map = HWC3(detected_map)
        img = resize_image(detected_map, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        print("222")
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        print("333")
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        print("444")
        if save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        print("555")
        cond = {
            "c_concat": [control], 
            "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        print("666")
        un_cond = {
            "c_concat": None if guess_mode else [control], 
            "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        print("777")
        shape = (4, H // 8, W // 8)
        print("888")
        if save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, _ = self.ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        print("999")
        if save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        print("1010")
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        print("1111")
        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results