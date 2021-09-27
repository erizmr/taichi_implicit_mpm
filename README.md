# Taichi Implicit MLS-MPM (Under construction)

Implicit MLS-MPM solved by Newton and CG. GGUI is required.

## Test
`python3 demo.py --dim 2 --implicit`
`python3 demo.py --dim 3 --implicit`

## Verify the gradient using difftest
`python3 demo.py --dim 2 --implicit --difftest`