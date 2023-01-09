def get_pipeline(
	model_name_or_path,
	vae_name_or_path=None, 
	tokenizer_name_or_path=None, 
	text_encoder_name_or_path=None,
	feature_extractor_name_or_path=None,
	scheduler=None,
	revision=None,
 	patch_dict,	
 	device="cuda",
):
	torch_dtype = torch.float16 if device == "cuda" else torch.float32

	if scheduler is None:
	    #scheduler = DPMSolverMultistepScheduler.from_pretrained(
	    #model_name_or_path, 
	    #subfolder="scheduler")

	    scheduler = DPMSolverMultistepScheduler(
	        beta_start=0.00085,
	        beta_end=0.012,
	        beta_schedule="scaled_linear",
	        num_train_timesteps=1000,
	        trained_betas=None,
	        prediction_type="epsilon",
	        thresholding=False,
	        algorithm_type="dpmsolver++",
	        solver_type="midpoint",
	        lower_order_final=True,
	    )

    pipe = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        custom_pipeline="lpw_stable_diffusion",
        safety_checker=None,
        revision=revision,
        scheduler=scheduler,
        vae=AutoencoderKL.from_pretrained(
            vae_name_or_path or model_name_or_path,
            subfolder=None if vae_name_or_path else "vae",
            revision=None if vae_name_or_path else revision,
            torch_dtype=torch_dtype,
        ),
        feature_extractor=feature_extractor_name_or_path,
        torch_dtype=torch_dtype
    )

    if patch_dict:
    	patch_pipeline(pipe, patch_dict)

    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

# def patch_pipeline(
# 	pipeline,
# 	patch_dict,
# ):
# 	if patch_dict["embeddings"]:
# 		# model after load_learned_embed_in_clip
# 	if patch_dict["text_encoder"]:
# 		# use load_state_dict?
# 	if patch_dict["text_encoder_loras"]:
# 		# inject/patch Loras, model after monkeypatch_or_replace_lora, make replace_module_w_lora
# 	if patch_dict["unet"]:
# 		# use load_state_dict?
# 	if patch_dict["unet_loras"]:
# 		# inject/patch Loras, model after monkeypatch_or_replace_lora
