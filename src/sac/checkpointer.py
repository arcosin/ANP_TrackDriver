import torch
import torch.nn as nn
import torch.optim as optim

import json
import os

class checkpointer:
    def __init__(self, model_class, save_dir, save_loc='.', model_params={}):
        self.model_class = model_class
        self.save_path = str(save_loc + '/' + save_dir)
        self.metadata = None
        if not os.path.isdir(save_loc):
            print('save loc does not exist')
        if not os.path.isdir(save_loc + '/' + save_dir):
            os.mkdir(self.save_path)
        if os.path.isfile(self.save_path + '/data.json')
            metafile = open(self.save_path + '/data.json', 'r')
            self.metadata = json.loads(metafile.read())
            metafile.close()
        # metafile = open(self.save_path + '/data.json', 'w')
        else:
            metadata = {
                'hyperparameters' : model_params,
                'checkpoints': [],
                'checkpoint_scores': [],
                'last_checkpoint_id': 0
            }
            metafile = open(self.save_path + '/data.json', 'w')
            metafile.write(json.dumps(metadata))
            metafile.close()

    def save_checkpoint(self, model, score=0):
        checkpoint_id = metadata['last_checkpoint_id'] + 1
        metadata['last_checkpoint_id'] += 1
        metadata['checkpoints'].append(checkpoint_id)
        metadata['checkpoint_scores'].append(score)
        torch.save({
            'fe_model_state_dict': model.fe.state_dict(),
            'value_model_state_dict': model.v_net.state_dict(),
            'target_value_model_state_dict': model.target_v_net.state_dict(),
            'q1_model_state_dict': model.q_net1.state_dict(),
            'q2_model_state_dict': model.q_net2.state_dict(),
            'pi_model_state_dict': model.pi_net.state_dict()
            'value_optimizer_state_dict': model.value_optimizer.state_dict(),
            'q1_optimizer_state_dict': model.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': model.q2_optimizer.state_dict(),
            'policy_optimizer_state_dict': model.policy_optimizer.state_dict(),
            'score': score,
            }, str(self.save_path + '/checkpoint_' + str(checkpoint_id))
        self.flush_metadata()

    def load_last_checkpoint(self, env):
        # TODO get correct model initializing setup
        checkpoint_id = metadata['checkpoints'][-1]
        return self.load_checkpoint(checkpoint_id, env)

    def get_checkpoints(self):
        return self.metadata['checkpoints'], self.metadata['checkpoint_scores']

    def load_checkpoint(self, checkpoint_id, env):
        model = self.model_class(env, **metadata['hyperparameters'])

        checkpoint = torch.load(str(self.save_path + '/checkpoint_' + str(checkpoint_id)))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        model.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        model.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        model.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        score = checkpoint['score']

    def delete_checkpoint(self, checkpoint_id):
        # TODO
        pass

    def flush_metadata(self):
        metafile = open(self.save_path + '/data.json', 'r')
        self.metadata = json.loads(metafile.read())
        metafile.close()
