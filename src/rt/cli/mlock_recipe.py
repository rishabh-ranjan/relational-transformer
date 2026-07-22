import tyro

from rt.mlock_recipe import MlockConfig, main


def default_config() -> MlockConfig:
    return MlockConfig(
        pre_dir=tyro.MISSING,
        include_dbs_file=None,
        embedding_model_ref="all-MiniLM-L12-v2",
        workers=32,
    )


if __name__ == "__main__":
    main(tyro.cli(MlockConfig, default=default_config()))
