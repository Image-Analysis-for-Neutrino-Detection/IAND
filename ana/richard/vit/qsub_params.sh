vit_dim=256
vit_depth=12
vit_heads=16
vit_mlp_dim=256
vit_dropout=0.1
vit_emb_dropout=0.1
tag='params_standard'
qsub -v vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,\
    vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout, \
    tag=$tag \
    run_vit_regression_simple_power_images_PHI_params.sh
#
vit_dim=512
vit_depth=12
vit_heads=16
vit_mlp_dim=256
vit_dropout=0.1
vit_emb_dropout=0.1
tag='params_test1'
qsub -v vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout,tag=$tag run_vit_regression_simple_power_images_PHI_params.sh
#
vit_dim=256
vit_depth=24
vit_heads=16
vit_mlp_dim=256
vit_dropout=0.1
vit_emb_dropout=0.1
tag='params_test2'
PARAMS=(vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,\
vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout,\
tag=$tag)
echo "${PARAMS[@]}" 
qsub -v "${PARAMS[@]}" run_vit_regression_simple_power_images_PHI_params.sh
#
vit_dim=256
vit_depth=12
vit_heads=32
vit_mlp_dim=256
vit_dropout=0.1
vit_emb_dropout=0.1
tag='params_test3'
PARAMS=(vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,\
vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout,\
tag=$tag)
echo "${PARAMS[@]}" 
qsub -v "${PARAMS[@]}" run_vit_regression_simple_power_images_PHI_params.sh

#
vit_dim=256
vit_depth=12
vit_heads=16
vit_mlp_dim=512
vit_dropout=0.1
vit_emb_dropout=0.1
tag='params_test4'
PARAMS=(vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,\
vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout,\
tag=$tag)
echo "${PARAMS[@]}" 
qsub -v "${PARAMS[@]}" run_vit_regression_simple_power_images_PHI_params.sh

#
vit_dim=256
vit_depth=12
vit_heads=16
vit_mlp_dim=256
vit_dropout=0.05
vit_emb_dropout=0.1
tag='params_test5'
PARAMS=(vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,\
vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout,\
tag=$tag)
echo "${PARAMS[@]}" 
qsub -v "${PARAMS[@]}" run_vit_regression_simple_power_images_PHI_params.sh

#
vit_dim=256
vit_depth=12
vit_heads=16
vit_mlp_dim=256
vit_dropout=0.1
vit_emb_dropout=0.05
tag='params_test6'
PARAMS=(vit_dim=$vit_dim,vit_depth=$vit_depth,vit_heads=$vit_heads,\
vit_mlp_dim=$vit_mlp_dim,vit_dropout=$vit_dropout,vit_emb_dropout=$vit_emb_dropout,\
tag=$tag)
echo "${PARAMS[@]}" 
qsub -v "${PARAMS[@]}" run_vit_regression_simple_power_images_PHI_params.sh

#
