# download aot-ckpt 
gdown --id '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./external/segment_and_track_anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

# download sam-ckpt
wget -P ./external/segment_and_track_anything/ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# download grounding-dino ckpt
wget -P ./external/segment_and_track_anything/ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# download mit-ast-finetuned ckpt
#wget -O ./ast_master/pretrained_models/audio_mdl.pth https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1