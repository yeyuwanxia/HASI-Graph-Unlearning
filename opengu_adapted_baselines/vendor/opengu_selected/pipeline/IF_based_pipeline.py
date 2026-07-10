import torch
import numpy as np
from task import get_trainer
# from memory_profiler import profile
BLUE_COLOR = "\033[34m"
RESET_COLOR = "\033[0m"
class IF_based_pipeline:
    """
    Base class for implementing an IF-based pipeline. This class defines the basic structure 
    and essential methods that must be implemented by subclasses. If you do not want to 
    implement the methods from scratch, please inherit from one of the derived classes 
    that extend this base class.
    
    Class Attributes:
        args (dict): A dictionary containing configuration arguments for the pipeline, including the number of unlearned nodes/edges, 
                     number of runs, downstream tasks, and other relevant settings.

        logger (Logger): A logger object for logging information during pipeline execution.

        data (Dataset): A dataset object that provides the data for the pipeline.

        model_zoo (ModelZoo): A model zoo that provides models and related functionality.

        num_runs (int): The number of runs to execute.

        run (int): The current run index.

        num_shards (int): The number of shards (partitions) in the pipeline.

        poison_f1 (np.ndarray): Array to store the poison F1 score for each run.

        average_f1 (np.ndarray): Array to store the average F1 score for each run.

        average_auc (np.ndarray): Array to store the average AUC score for each run.

        avg_partition_time (np.ndarray): Array to store the average partition time for each run.

        avg_training_time (np.ndarray): Array to store the average training time for each run.

        avg_unlearning_time (np.ndarray): Array to store the average unlearning time for each run.

        deleted_nodes (np.ndarray): Array to store the deleted nodes.

        feature_nodes (np.ndarray): Array to store the feature nodes.

        influence_nodes (np.ndarray): Array to store the influence nodes.

        deleted_edges (np.ndarray): Array to store the deleted edges.

        influence_edges (np.ndarray): Array to store the influence edges.

        num_feats (int): The number of features in the dataset.
    """
    def __init__(self,args,logger,model_zoo):
        """
        Initializes the IF_based_pipeline with the provided arguments, logger, and model zoo.

        Args:
            args (dict): A dictionary containing the configuration parameters.

            logger (Logger): A logger object used to log runtime information.

            model_zoo (ModelZoo): An object that provides access to models and datasets.
        """
        self.args = args
        self.logger = logger
        self.data = model_zoo.data
        self.model_zoo = model_zoo
        self.num_runs = self.args["num_runs"]
        self.run = 0
        self.num_shards = self.args["num_shards"]
        self.poison_f1 = np.zeros(self.args["num_runs"])
        self.average_f1 = np.zeros(self.args["num_runs"])
        self.average_auc = np.zeros(self.args["num_runs"])
        self.avg_partition_time = np.zeros(self.args["num_runs"])
        self.avg_training_time = np.zeros(self.args["num_runs"])
        self.avg_unlearning_time = np.zeros(self.args["num_runs"])
        # self.training_time = np.zeros(self.num_runs)
        self.deleted_nodes = np.array([])
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])
        self.deleted_edges = np.array([])
        self.influence_edges = np.array([])
        self.num_feats = self.data.num_features
    
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
            self.train_original_model(self.run)
            self.unlearning_request()
            self.unlearn()
            self.logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024/1024}MB")
            self.logger.info(f"Max Cached: {torch.cuda.max_memory_reserved()/1024/1024}MB")

    def run_exp(self):
        """
        Run the experimental process for multiple iterations, training and unlearning the model.

        This method runs the experiment for a specified number of iterations (`num_runs`).
        During each run, it:

        1. Initializes the target model using the base model provided in `args`.
        2. Trains the original model.
        3. Executes the unlearning request and performs the unlearning process.
        4. If the `unlearn_task` and `downstream_task` are set to "node", it triggers the MIA (Model Inversion Attack) for nodes.

        After all runs, it logs the performance metrics, including:
        
        - Poison F1 Score
        - Unlearn F1 Score
        - Average AUC Score
        - Average Training Time
        - Average Unlearning Time

        Args:
            self: The instance of the class that contains the experiment configuration and data.

        """
        for self.run in range(self.args['num_runs']):
            self.target_model_name = self.args['base_model']
            self.determine_target_model()
            self.train_original_model(self.run)
            self.unlearning_request()
            self.unlearn()
            
            
            if self.args["unlearn_task"]=="node" and self.args["downstream_task"]=="node":
                self.mia_attack()
            # elif self.args["unlearn_task"]=="edge":
            #     self.mia_attack_edge()
                
        
        self.logger.info(
        "{}Performance Metrics:\n"
        " - Poison F1 Score: {:.4f} ± {:.4f}\n"
        " - Unlearn F1 Score: {:.4f} ± {:.4f}\n"
        " - Average AUC Score: {:.4f} ± {:.4f}\n"
        " - Average Training Time: {:.4f} ± {:.4f}\n"
        " - Average Unlearning Time: {:.4f} ± {:.4f} seconds{}".format(
            BLUE_COLOR,
            np.mean(self.poison_f1), np.std(self.poison_f1),
            np.mean(self.average_f1), np.std(self.average_f1),
            np.mean(self.average_auc), np.std(self.average_auc),
            np.mean(self.avg_training_time), np.std(self.avg_training_time),
            np.mean(self.avg_unlearning_time), np.std(self.avg_unlearning_time),
            RESET_COLOR
            )
        )
        pass
    
    def unlearning_request(self):
        pass
    
    def determine_target_model(self):
        # self.args["unlearn_trainer"] = trainer
        # self.target_model = get_trainer(self.args,self.logger,self.model_zoo.model,self.data)
        pass
    def train_original_model(self,run):
        pass
    
    def approxi(self,result_tuple):
        pass
    
    def mia_attack(self):
        pass
    
    def get_if_grad(self,run):
        pass

    def mia_attack_edge(self):
        pass
    
    def unlearn(self):
        pass
    