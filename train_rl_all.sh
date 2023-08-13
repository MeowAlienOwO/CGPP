# python -m bpp1d.main rl train -d or-gym-lw0 -C 9 --path models/rl_lw0_9 --epoch 500
# python -m bpp1d.main rl train -d or-gym-bw0 -C 9 --path models/rl_bw0_9 --epoch 500
# python -m bpp1d.main rl train -d or-gym-pp0 -C 9 --path models/rl_pp0_9 --epoch 500
# python -m bpp1d.main rl train -d uniform --ir 1 --ir 100 -C 100 --path models/rl_uniform_range100_100 --epoch 500
# python -m bpp1d.main rl train -d or-gym-lw1 -C 9 --path models/rl_lw1_9 --epoch 500
# python -m bpp1d.main rl train -d or-gym-bw1 -C 9 --path models/rl_bw1_9 --epoch 500
# python -m bpp1d.main rl train -d or-gym-pp1 -C 9 --path models/rl_pp1_9 --epoch 500


python -m bpp1d.main rl train -d or-gym-lw1 -C 100 --path models/rl_lw1_cap100_100 --epoch 500
python -m bpp1d.main rl train -d or-gym-pp1 -C 100 --path models/rl_pp1_cap100_100 --epoch 500
# python -m bpp1d.main rl train -d or-gym-bw1 -C 100 --path models/rl_bw1_cap100_100 --epoch 500

python -m bpp1d.main rl train -d uniform --ir 150 --ir 200 -C 1000 --path models/rl_uniform_range200_1000 --epoch 500