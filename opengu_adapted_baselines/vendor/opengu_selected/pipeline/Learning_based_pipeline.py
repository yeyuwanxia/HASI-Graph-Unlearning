import numpy as np
# from memory_profiler import profile
import torch
BLUE_COLOR = "\033[34m"
RESET_COLOR = "\033[0m"
class Learning_based_pipeline:
    """
    Base class for implementing a learning-based pipeline. This class defines the basic structure
    and essential methods that must be implemented by subclasses. It provides attributes for managing
    runs, data, and performance metrics.

    Class Attributes:
        args (dict): A dictionary containing configuration arguments for the pipeline.

        logger (Logger): A logger object for logging information during pipeline execution.

        data (Dataset): A dataset object that provides the data for the pipeline.

        model_zoo (ModelZoo): A model zoo that provides models and related functionality.

        run (int): The current run index.

        num_shards (int): The number of shards (partitions) in the pipeline.

        poison_f1 (np.ndarray): Array to store the poison F1 score for each run.

        average_f1 (np.ndarray): Array to store the average F1 score for each run.

        average_auc (np.ndarray): Array to store the average AUC score for each run.

        avg_training_time (np.ndarray): Array to store the average training time for each run.

        avg_unlearning_time (np.ndarray): Array to store the average unlearning time for each run.

        avg_sampling_time (np.ndarray): Array to store the average sampling time for each run.
    """
    def __init__(self,args,logger,model_zoo):
        """
        Initializes the Learning_based_pipeline with the provided arguments, logger, and model zoo.

        Args:
            args (dict): A dictionary containing the configuration parameters. It must include keys like "num_runs" and "num_shards".
            logger (Logger): A logger object used to log runtime information.
            model_zoo (ModelZoo): An object that provides access to models and datasets.
        """
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.run = 0
        self.num_shards = self.args["num_shards"]
        self.poison_f1 = np.zeros(self.args["num_runs"])
        self.average_f1 = np.zeros(self.args["num_runs"])
        self.average_auc = np.zeros(self.args["num_runs"])
        self.avg_training_time = np.zeros(self.args["num_runs"])
        self.avg_unlearning_time = np.zeros(self.args["num_runs"])
        self.avg_sampling_time = np.zeros(self.args["num_runs"])
    
    # @profile
    def run_exp_mem(self):
        """
        Executes the experimental pipeline while profiling memory usage.

        During each run, this method:

        1. Seeds the random number generator for reproducibility.
        2. Executes the partitioning step.
        3. Trains the shard-based models.
        4. Performs the unlearning step.

        """
        for self.run in range(self.args["num_runs"]):
            self.determine_target_model()
            self.train_original_model()
            self.unlearning_request()
            self.unlearn()
            self.logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024/1024}MB")
            self.logger.info(f"Max Cached: {torch.cuda.max_memory_reserved()/1024/1024}MB")

    def run_exp(self):
        """
        Executes the experimental pipeline for multiple runs, performing training, unlearning, and evaluation.

        During each run, this method:

        1. Determines the target model.
        2. Trains the original model.
        3. Processes unlearning requests.
        4. Performs unlearning operations.
        5. Conducts attacks based on the unlearning task (node or edge).

        At the end of all runs, logs the performance metrics including:

        - Poison F1 Score
        - Unlearn F1 Score
        - Average AUC Score
        - Average Training Time
        - Average Sampling Time
        - Average Unlearning Time

        """
        for self.run in range(self.args['num_runs']):
            self.determine_target_model()
            self.train_original_model()
            self.unlearning_request()
            self.unlearn()
            if self.args["downstream_task"] == "node" and self.args["unlearn_task"]=="node" and False:
                self.mia_attack()
            # elif self.args["unlearn_task"]=="edge":
            #     self.mia_attack_edge()
        self.logger.info(
        "{}Performance Metrics:\n"
        " - Poison F1 Score: {:.4f} ± {:.4f}\n"
        " - Unlearn F1 Score: {:.4f} ± {:.4f}\n"
        " - Average AUC Score: {:.4f} ± {:.4f}\n"
        " - Average Training Time: {:.4f} ± {:.4f}\n"
        " - Average Sampling Time: {:.4f} ± {:.4f} seconds\n"
        " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
            BLUE_COLOR,
            np.mean(self.poison_f1), np.std(self.poison_f1),
            np.mean(self.average_f1), np.std(self.average_f1),
            np.mean(self.average_auc), np.std(self.average_auc),
            np.mean(self.avg_training_time), np.std(self.avg_training_time),
            np.mean(self.avg_sampling_time), np.std(self.avg_sampling_time),
            np.mean(self.avg_unlearning_time), np.std(self.avg_unlearning_time),
            RESET_COLOR
            )
        )
            
    def determine_target_model(self):
        pass
            
    def train_original_model(self):
        pass
            
    def unlearning_request(self):
        pass
    
    def unlearn(self):
        pass
    
    def mia_attack(self):
        pass
    
    def mia_attack_edge(self):
        pass