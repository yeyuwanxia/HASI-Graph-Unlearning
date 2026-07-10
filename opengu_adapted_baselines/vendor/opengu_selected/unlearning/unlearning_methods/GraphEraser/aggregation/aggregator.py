import logging
import torch

torch.cuda.empty_cache()
from utils import dataset_utils
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import numpy as np
import time
from torch_geometric.loader import DataLoader
from unlearning.unlearning_methods.GraphEraser.aggregation.optimal_aggregator import OptimalAggregator
from unlearning.unlearning_methods.GraphEraser.aggregation.optimal_edge_aggregator import OptimalEdgeAggregator
from dataset.original_dataset import original_dataset
from utils.dataset_utils import *
from unlearning.unlearning_methods.GraphEraser.aggregation.contra_aggregator_v2 import ContrastiveAggregator

class Aggregator:
    def __init__(self, run, target_model, data, shard_data, args,logger,affected_shard=None):
        self.logger = logger
        self.args = args

        self.data_store = original_dataset(self.args,logger)

        self.run = run
        self.target_model =target_model
        self.data = data
        self.shard_data = shard_data
        self.affected_shard = affected_shard
        self.num_shards = args['num_shards']
        

    def generate_posterior(self, suffix=""):
        
        if self.args["downstream_task"] == "node":
            self.true_label = self.shard_data[0].y[self.shard_data[0]['test_mask']].detach().cpu().numpy()
            self.posteriors = {}
            if self.args['aggregator'] == 'contrastive':
                self.posteriors = []
            self.test_embeddings = []
            for shard in range(self.args['num_shards']):
                self.target_model.data = self.shard_data[shard]
                if self.affected_shard is not None and shard in self.affected_shard:
                    load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
                else:
                    load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")
                
                if self.args['aggregator'] == 'contrastive':
                    z, f = self.target_model.posterior_con(return_features=True)
                    self.posteriors.append(z)
                    #self.test_embeddings.append(torch.cat([f, z], 1))
                    self.test_embeddings.append(f)
                else:
                    if self.args["downstream_task"]=="node":
                        self.posteriors[shard] = self.target_model.posterior()
            if self.args['aggregator'] == 'contrastive':        
                self.posteriors = torch.stack(self.posteriors) 
                if len(self.test_embeddings):
                    self.test_embeddings = torch.stack(self.test_embeddings)
            self.logger.info("Saving posteriors.")
            save_posteriors(self.logger,self.args,self.posteriors, self.run, suffix)
                    
                    
        elif self.args["downstream_task"] == "edge":
            pos_edge_labels = torch.ones(self.shard_data[0].test_edge_index.size(1),dtype=torch.float32)
            neg_edge_labels = torch.zeros(self.shard_data[0].test_edge_index.size(1),dtype=torch.float32)
            edge_labels = torch.cat((pos_edge_labels,neg_edge_labels))
            self.true_label = edge_labels
            self.posteriors = {}
            if self.args['aggregator'] == 'contrastive':
                self.posteriors = []
            self.test_embeddings = []
            for shard in range(self.args['num_shards']):
                self.target_model.data = self.shard_data[shard]
                if self.affected_shard is not None and shard in self.affected_shard:
                    load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
                else:
                    load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")

                if self.args['aggregator'] == 'contrastive':
                    z, f = self.target_model.posterior_con(return_features=True)
                    self.posteriors.append(z)
                    #self.test_embeddings.append(torch.cat([f, z], 1))
                    self.test_embeddings.append(f)
                else:
                    if self.args["downstream_task"]=="node":
                        self.posteriors[shard] = self.target_model.posterior()
                    elif self.args["downstream_task"]=="edge":
                        self.posteriors[shard] = self.target_model.posterior_edge()
                        
            if self.args['aggregator'] == 'contrastive':        
                self.posteriors = torch.stack(self.posteriors) 
                if len(self.test_embeddings):
                    self.test_embeddings = torch.stack(self.test_embeddings)
            self.logger.info("Saving posteriors.")
            save_posteriors(self.logger,self.args,self.posteriors, self.run, suffix)
        else:
            self.posteriors = []
            self.true_label = []
            test_loader = DataLoader(self.shard_data[0][1], batch_size=64, shuffle=False)
            for tmpdata in test_loader:
                self.true_label.append(tmpdata.y)
            for shard in range(self.args['num_shards']):
                graph_data = self.shard_data[shard]
                if self.affected_shard is not None and shard in self.affected_shard:
                    load_target_model(self.logger,self.args,self.run, self.target_model, shard, "_unlearned")
                else:
                    load_target_model(self.logger, self.args, self.run, self.target_model, shard, "")
                test_loader = DataLoader(graph_data[1], batch_size=64, shuffle=False)
                tmp_list = []
                for tmpdata in test_loader:
                    tmpdata = tmpdata.cuda()
                    tmp_list.append(self.target_model.model(tmpdata.x, tmpdata.edge_index,batch = tmpdata.batch))
                posteriors = torch.cat(tmp_list)
                posteriors =  posteriors.squeeze(0) 
                self.posteriors.append(posteriors)
            self.posteriors = torch.stack(self.posteriors,dim = 0)
                
            save_posteriors(self.logger,self.args,self.posteriors, self.run, suffix)
                

    def aggregate(self,data):
        if self.args['aggregator'] == 'mean':
            aggregate_f1_score = self._mean_aggregator(data)
        elif self.args['aggregator'] == 'optimal':
            aggregate_f1_score = self._optimal_aggregator(data)
        elif self.args['aggregator'] == 'majority':
            aggregate_f1_score = self._majority_aggregator(data)
        elif self.args['aggregator'] == 'contrastive':
            aggregate_f1_score, t = self._contrastive_aggregator()
        else:
            raise Exception("unsupported aggregator.")

        return aggregate_f1_score

    def _mean_aggregator(self,data):
        posterior = self.posteriors[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard]

        posterior = posterior / self.num_shards
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, posterior.detach().cpu().numpy(),average="micro")
        elif self.args["downstream_task"]=="graph":
            posterior = posterior.detach().cpu()
            pred = posterior.argmax(dim=1)
            self.true_label = torch.concat(self.true_label,dim=0).cpu()
            return accuracy_score(self.true_label,pred)

    def _majority_aggregator(self,data):
        pred_labels = []
        for shard in range(self.num_shards):
            edge_pred = torch.where(self.posteriors[shard] > 0.5, torch.tensor(1), torch.tensor(0))
            pred_labels.append(edge_pred.cpu().numpy())
        pred_labels = np.stack(pred_labels)
        pred_label = np.argmax(
            np.apply_along_axis(np.bincount, axis=0, arr=pred_labels, minlength=self.posteriors[0].shape[0]), axis=0)
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, pred_label, average="micro")
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, pred_label, average="micro")
        elif self.args["downstream_task"]=="graph":
            return accuracy_score(self.true_label, pred_label, average="micro")

    def _optimal_aggregator(self,data):
        if self.args["downstream_task"]=="node":
            optimal = OptimalAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        elif self.args["downstream_task"]=="edge":
            optimal = OptimalEdgeAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        optimal.generate_train_data(data)
        weight_para = optimal.optimization()
        save_optimal_weight(self.logger,self.args, weight_para, run=self.run)

        posterior = self.posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            # print(self.posteriors[shard],weight_para[shard])
            posterior += self.posteriors[shard] * weight_para[shard]
        
        # print(posterior,self.true_label_edge)
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, posterior.detach().cpu().numpy(), average="micro")

    def _contrastive_aggregator(self):
        proj = ContrastiveAggregator(self.run, self.target_model, self.data, self.args,self.logger)
        proj._generate_train_data()
        
        start_time = time.time()
        proj_model = proj.optimization()#.to(self.posteriors.device)
        proj_model.eval()
        if self.args['base_model'] == 'GIN':
            self.test_embeddings = torch.tanh(self.test_embeddings)
        self.test_embeddings = self.test_embeddings.permute(1, 0, 2).to(next(proj_model.parameters()).device)
        #start_time = time.time()
        posterior = proj_model(self.test_embeddings, is_eval=True)
        aggr_time = time.time() - start_time
        
        dataset_utils.save_optimal_weight(self.logger,self.args,proj_model, run=self.run)
        if self.args["downstream_task"]=="node":
            return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), aggr_time
        elif self.args["downstream_task"]=="edge":
            posterior = torch.where(posterior > 0.5, torch.tensor(1), torch.tensor(0))
            return roc_auc_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), aggr_time