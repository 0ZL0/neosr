# neosr Debug Fork

This repository is a **non-official debug fork** of [neosr](https://github.com/neosr-project/neosr).
It is intended for local debugging, architecture integration, and experiment work. It may contain temporary changes, incomplete integrations, or behavior that differs from upstream.

For official documentation, installation guidance, configuration details, and other related information, please refer to the official [neosr](https://github.com/neosr-project/neosr) project.

## Quick Start

Train:

```bash
neosr-train config.toml
```

Test:

```bash
neosr-test config.toml
```

Configuration templates are available in [options](options/).

## Notes

- `pftsr` / `pftsr_light` strongly depend on `smm_cuda` to achieve normal performance. Without it, inference may consume very large compute and VRAM resources. A local Windows build is available here for reference only: [smm_cuda-1.0-cp313-cp313-win_amd64.whl](wheels_for_windows/smm_cuda-1.0-cp313-cp313-win_amd64.whl). This wheel was built against Python 3.13, PyTorch 2.7.0+cu128, CUDA Toolkit 12.8, and MSVC 2022 on Windows.
- `dvmsr` depends on the `mamba_ssm` library.

## Supported Archs

| arch                                                                                              | option                                 			|
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                             | `esrgan`                               			|
| [SRVGGNetCompact](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/srvgg_arch.py) | `compact`                              			|
| [SwinIR](https://github.com/JingyunLiang/SwinIR)                                                  | `swinir_small`, `swinir_medium`        			|
| [HAT](https://github.com/XPixelGroup/HAT)                                                         | `hat_s`, `hat_m`, `hat_l`              			|
| [DVMSR](https://github.com/nathan66666/DVMSR)                                                     | `dvmsr`                                			|
| [PFT-SR](https://github.com/LabShuHangGU/PFT-SR)                                                  | `pftsr`, `pftsr_light`                 			|
| [OmniSR](https://github.com/Francis0625/Omni-SR)                                                  | `omnisr`                               			|
| [SRFormer](https://github.com/HVision-NKU/SRFormer)                                               | `srformer_light`, `srformer_medium`    			|
| [DAT](https://github.com/zhengchen1999/dat)                                                       | `dat_small`, `dat_medium`, `dat_2` 			|
| [DITN](https://github.com/yongliuy/DITN)							    | `ditn`				     	      		|
| [DCTLSA](https://github.com/zengkun301/DCTLSA)						    | `dctlsa`							|
| [SPAN](https://github.com/hongyuanyu/SPAN)							    | `span`, `span_fast`					|
| [Real-CUGAN](https://github.com/bilibili/ailab)				  		    | `cugan`			   				|
| [CRAFT](https://github.com/AVC2-UESTC/CRAFT-SR)		 				    | `craft`			   				|
| [SAFMN](https://github.com/sunny2109/SAFMN)							    | `safmn`, `safmn_l`					|
| [RGT](https://github.com/zhengchen1999/RGT)							    | `rgt`, `rgt_s` 						|
| [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary)   				    | `atd`, `atd_light`   					|
| [PLKSR](https://github.com/dslisleedh/PLKSR)			     				    | `plksr`, `plksr_tiny`	    				|
| [RealPLKSR](https://github.com/dslisleedh/PLKSR/issues/4)					    | `realplksr`, `realplksr_s`, `realplksr_l`			|
| [DRCT](https://github.com/ming053l/DRCT)			     				    | `drct`, `drct_l`, `drct_s`   				|
| [MSDAN](https://github.com/Supereeeee/MSDAN)			     				    | `msdan`			    				|
| [SPANPlus](https://github.com/umzi2/SPANPlus)							    | `spanplus`, `spanplus_sts`, `spanplus_s`, `spanplus_st`   |
| [HiT-SRF](https://github.com/XiangZ-0/HiT-SR)							    | `hit_srf`, `hit_srf_medium`, `hit_srf_large`		|
| [HMA](https://github.com/korouuuuu/HMA)							    | `hma`, `hma_medium`, `hma_large`				|
| [MAN](https://github.com/icandle/MAN)								    | `man`, `man_tiny`, `man_light`				|
| [light-SAFMN++](https://github.com/sunny2109/SAFMN/tree/main/AIS2024-RTSR)			    | `light_safmnpp`						|
| [GRFormer](https://github.com/sisrformer/GRFormer)						    | `grformer`, `grformer_medium`, `grformer_large`  |
| [EIMN](https://github.com/liux520/EIMN)							    | `eimn`, `eimn_a`, `eimn_l`			   |
| [LMLT](https://github.com/jwgdmkj/LMLT)					| `lmlt`, `lmlt_tiny`, `lmlt_large`		   |
| [DCT](https://github.com/zhengchen1999/DCT)					| `dct`						   |
| [KRGN](https://github.com/ZhangDY827/KRGN)					| `krgn`					   |
| [PlainUSR](https://github.com/icandle/PlainUSR)				| `plainusr`, `plainusr_ultra`, `plainusr_large`   |
| [HASN](https://github.com/nathan66666/HASN)					| `hasn`					   |
| [FlexNet](https://github.com/umzi2/FlexNet)					| `flexnet`, `metaflexnet`			   |
| [CFSR](https://github.com/Aitical/CFSR)					| `cfsr`					   |
| [Sebica](https://github.com/idiosyncracies/Sebica)				| `sebica`, `sebica_mini`			   |
| [NinaSR](https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md)	| `ninasr`, `ninasr_b0`, `ninasr_b2`		   |
| [RCAN](https://github.com/yulunzhang/RCAN)					| `rcan`					   |
| [RFDN](https://github.com/njulj/RFDN)					| `rfdn`, `rfdn_l`, `rfdn_e`		   |
| [MoESR](https://github.com/umzi2/MoESR)					| `moesr`					   |
| [ASID](https://github.com/saturnian77/ASID)					| `asid`, `asid_d8`				   |
| [MoSRV2](https://github.com/umzi2/MoSRV2)					| `mosrv2`					   |
| [ESC](https://github.com/dslisleedh/ESC)					| `esc`, `esc_fp`, `esc_large`			   |
| [CATANet](https://github.com/EquationWalker/CATANet)				| `catanet`					   |
| [MFG-HMoE](https://github.com/Mr-Bamboo/MFG-HMoE)				| `mfghmoe`					   |
| [MicroSR](https://github.com/YanHuiGUO/Submission-for-NTIRE-2025-SR-x4-)	| `microsr`					   |

## License

Released under the [Apache license](license.txt).
Additional notices are listed in [licenses/readme.md](licenses/readme.md).
