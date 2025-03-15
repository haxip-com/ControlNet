def process(detected_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    print(num_samples, type(num_samples))
    with torch.no_grad():
        detected_map = HWC3(detected_map) # 以输入图片1020 x 1920为例，(h, w, c) （高，宽，channel）(1020, 1920, 3)
        img = resize_image(detected_map, image_resolution) # (1020, 1920, 3) -> (512, 960, 3)
        H, W, C = img.shape # 512, 960, 3
        print(11111)
        # cv2.INTER_NEAREST：最邻近插值
        # 使用 最邻近插值法 将图像 detected_map 调整为指定大小 (W, H)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0 # shape=torch.size([512,960,3]), dtype=torch.float32, 值在0-1区间
        control = torch.stack([control for _ in range(num_samples)], dim=0) # num_samples=1, shape=torch.size([1,512,960,3])
        control = einops.rearrange(control, 'b h w c -> b c h w').clone() # b h w c -> b c h w

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed) #随机种子

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        # !!!
        # obstruction_prompt="A construction zone occupies part of the right lane, with cones, workers, and safety signs."
        
        # !!!
        # cond（正向控制条件）：图片转成control，prompt+a_prompt作为交叉注意力的控制条件
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        # cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 
        #         "obstruction_c_crossattn": [model.get_learned_conditioning([obstruction_prompt] * num_samples)]}
        
        # !!!
        # un_cond（反向控制条件）：图片转成control，n_prompt作为交叉注意力的负面控制条件
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 
        #         "obstruction_c_crossattn": None}
        
        shape = (4, H // 8, W // 8) # H 512, W 960 -> shape (4, 64, 120)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # 控制权重
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # samples -> shape=torch.size([1,4,64,120])
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples) # shape=torch.size([1,3,512,960])
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)] # list
    return [detected_map] + results