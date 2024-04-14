use std::path::Path;

pub use burn;
use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    config::Config,
    data::dataset::{SqliteDatasetError, SqliteDatasetWriter},
    optim::AdamConfig,
};
pub use clap;
use serde::{de::DeserializeOwned, Serialize};

pub fn create_dataset<T>(
    gen: impl Fn() -> T + Sync,
    path: impl AsRef<Path>,
    train_len: usize,
    test_len: usize,
) -> Result<(), SqliteDatasetError>
where
    T: Serialize + Send + Clone + Sync + DeserializeOwned,
{
    let mut db = SqliteDatasetWriter::<T>::new(path, true)?;
    (0..train_len)
        .into_iter()
        .try_for_each::<_, Result<(), SqliteDatasetError>>(|_| {
            db.write("train", &gen())?;
            Ok(())
        })?;
    (0..test_len)
        .into_iter()
        .try_for_each::<_, Result<(), SqliteDatasetError>>(|_| {
            db.write("test", &gen())?;
            Ok(())
        })?;

    db.set_completed()
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

#[macro_export]
macro_rules! train {
    ($path: expr, $db_name: expr, $batcher: tt, $item: ty, $model_name: expr, $device: ident, $backend: ty, $model_config_path: expr, $model_config_ty: ty) => {{
        use $crate::burn::{
            config::Config,
            data::{dataloader::DataLoaderBuilder, dataset::SqliteDataset},
            module::Module,
            prelude::Backend,
            record::CompactRecorder,
            train::{
                metric::{AccuracyMetric, LossMetric},
                LearnerBuilder,
            },
        };
        use $crate::TrainingConfig;

        let path: &Path = $path.as_ref();
        let models_path = path.join("models");
        let config: TrainingConfig = TrainingConfig::load(path.join("training.json"))
            .expect("Training config should be loaded successfully");
        std::fs::create_dir_all(&models_path)
            .expect("models directory should be created successfully");
        let db_path = path.join($db_name);
        <$backend>::seed(config.seed);

        let batcher_train = <$batcher<DefaultAutodiffBackend>>::new($device.clone());
        let batcher_valid = <$batcher<DefaultBackend>>::new($device.clone());

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(
                SqliteDataset::<$item>::from_db_file(&db_path, "train")
                    .expect("Dataset should be read successfully"),
            );

        let dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(
                SqliteDataset::<$item>::from_db_file(&db_path, "test")
                    .expect("Dataset should be read successfully"),
            );

        let model_config = <$model_config_ty>::load($model_config_path)
            .expect("Model config should be loaded successfully");

        let learner = LearnerBuilder::new($path)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .devices(vec![$device.clone()])
            .num_epochs(config.num_epochs)
            .summary()
            .build(
                model_config.init::<$backend>(&$device),
                config.optimizer.init(),
                config.learning_rate,
            );

        let model_trained = learner.fit(dataloader_train, dataloader_test);

        model_trained
            .save_file(models_path.join("model.mpk"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");

        let mut renamed = models_path.join($model_name);
        renamed.set_extension("mpk");

        std::fs::rename(models_path.join("model.mpk"), renamed)
            .expect("Model should be renamed successfully");
    }};
}

pub type DefaultBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type DefaultAutodiffBackend = Autodiff<DefaultBackend>;

#[macro_export]
macro_rules! ml_app {
    ($batcher: tt, $item: ty, $gen: expr, $model_config_ty: ty) => {{
        use std::path::Path;
        use $crate::burn::backend::{
            wgpu::{AutoGraphicsApi, WgpuDevice},
            Autodiff, Wgpu,
        };
        use $crate::clap::{self, command, Parser, Subcommand};
        use $crate::{create_dataset, train, DefaultAutodiffBackend, DefaultBackend};

        #[derive(Parser)]
        #[command(author, version, about, long_about = None)]
        struct Cli {
            #[command(subcommand)]
            command: Commands,
        }

        #[derive(Subcommand)]
        enum Commands {
            Gen { train_len: usize, test_len: usize },
            Train,
            Test,
        }

        let cli = Cli::parse();

        let artifact_dir = format!("{}-academy", env!("CARGO_PKG_NAME"));
        std::fs::create_dir_all(&artifact_dir)
            .expect("Artifact directory should be created successfully");

        match cli.command {
            Commands::Gen {
                train_len,
                test_len,
            } => {
                create_dataset(
                    $gen,
                    Path::new(&artifact_dir).join("dataset.db"),
                    train_len,
                    test_len,
                )
                .expect("Dataset should be created successfully");
            }
            Commands::Train => {
                let device = $crate::burn::backend::wgpu::WgpuDevice::default();
                train!(
                    &artifact_dir,
                    "dataset.db",
                    $batcher,
                    $item,
                    "model",
                    device,
                    DefaultAutodiffBackend,
                    Path::new(&artifact_dir).join("model.json"),
                    $model_config_ty
                );
            }
            Commands::Test => {
                let device = $crate::burn::backend::wgpu::WgpuDevice::default();
            }
        }
    }};
}
