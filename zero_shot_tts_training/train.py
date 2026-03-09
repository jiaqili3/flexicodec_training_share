from omegaconf import DictConfig, OmegaConf
from typing import Optional
import hydra
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.append(f'{str(Path(__file__).parent.parent)}')

def train(cfg):
    if hasattr(cfg.trainer, 'trainer'):
        trainer = hydra.utils.instantiate(cfg.trainer.trainer)
    else:
        trainer = hydra.utils.instantiate(cfg.trainer)

    if hasattr(cfg.trainer, 'dataloader') and cfg.trainer.dataloader is not None:
        trainer._build_dataloader(
            hydra.utils.instantiate(cfg.trainer.dataloader)
        )
    else:
        trainer._build_dataloader(
            hydra.utils.instantiate(cfg.data.dataloader)
        )
    # test dataloder
    # dl = trainer.train_dataloader
    # breakpoint()
    # for batch in dl:
    #     print(batch)
    #     breakpoint()
    trainer.train_loop()


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="codec_train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:    
    # train the model
    train(cfg)


if __name__ == "__main__":
    main(None)
