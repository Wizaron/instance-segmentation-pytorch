import os
import time
import visdom
from sklearn.manifold import TSNE
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import ifilter

from cvppp_arch import Architecture as CVPPPArchitecture
from losses import DiceLoss, DiceCoefficient, DiscriminativeLoss


class Model(object):

    def __init__(
            self,
            dataset,
            n_classes,
            max_n_objects,
            use_instance_segmentation=False,
            use_coords=False,
            load_model_path='',
            usegpu=True):

        self.dataset = dataset
        self.n_classes = n_classes
        self.max_n_objects = max_n_objects
        self.use_instance_segmentation = use_instance_segmentation
        self.use_coords = use_coords
        self.load_model_path = load_model_path
        self.usegpu = usegpu

        assert self.dataset in ['CVPPP', ]

        if self.dataset == 'CVPPP':
            self.model = CVPPPArchitecture(
                self.n_classes,
                self.use_instance_segmentation,
                self.use_coords,
                usegpu=self.usegpu)

        self.__load_weights()

        if self.usegpu:
            cudnn.benchmark = True
            self.model.cuda()
            # self.model = torch.nn.DataParallel(self.model,
            #                                    device_ids=range(self.ngpus))

        print self.model

        self.vis = visdom.Visdom()
        self.training_metric_vis, self.test_metric_vis = None, None
        if self.use_instance_segmentation:
            self.instance_seg_vis = None

    def __load_weights(self):

        if self.load_model_path != '':
            assert os.path.isfile(self.load_model_path), 'Model : {} does not \
                exists!'.format(self.load_model_path)
            print 'Loading model from {}'.format(self.load_model_path)

            model_state_dict = self.model.state_dict()

            if self.usegpu:
                pretrained_state_dict = torch.load(self.load_model_path)
            else:
                pretrained_state_dict = torch.load(
                    self.load_model_path, map_location=lambda storage,
                    loc: storage)

            model_state_dict.update(pretrained_state_dict)
            self.model.load_state_dict(model_state_dict)

    def __define_variable(self, tensor, volatile=False):
        return Variable(tensor, volatile=volatile)

    def __define_input_variables(
            self, features, fg_labels, ins_labels, n_objects, mode):

        volatile = True
        if mode == 'training':
            volatile = False

        features_var = self.__define_variable(features, volatile=volatile)
        fg_labels_var = self.__define_variable(fg_labels, volatile=volatile)
        ins_labels_var = self.__define_variable(ins_labels, volatile=volatile)
        n_objects_var = self.__define_variable(n_objects, volatile=volatile)

        return features_var, fg_labels_var, ins_labels_var, n_objects_var

    def __define_criterion(self, class_weights, delta_var,
                           delta_dist, norm=2, optimize_bg=False,
                           criterion='CE'):
        assert criterion in ['CE', 'Dice', 'Multi', None]

        smooth = 1.0

        # Discriminative Loss
        if self.use_instance_segmentation:
            self.criterion_discriminative = DiscriminativeLoss(
                delta_var, delta_dist, norm, self.usegpu)
            if self.usegpu:
                self.criterion_discriminative = \
                    self.criterion_discriminative.cuda()

        # FG Segmentation Loss
        if class_weights is not None:
            class_weights = self.__define_variable(
                torch.FloatTensor(class_weights))
            if criterion in ['CE', 'Multi']:
                self.criterion_ce = torch.nn.CrossEntropyLoss(class_weights)
            if criterion in ['Dice', 'Multi']:
                self.criterion_dice = DiceLoss(
                    optimize_bg=optimize_bg, weight=class_weights,
                    smooth=smooth)
        else:
            if criterion in ['CE', 'Multi']:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
            if criterion in ['Dice', 'Multi']:
                self.criterion_dice = DiceLoss(
                    optimize_bg=optimize_bg, smooth=smooth)

        # MSE Loss
        self.criterion_mse = torch.nn.MSELoss()

        if self.usegpu:
            if criterion in ['CE', 'Multi']:
                self.criterion_ce = self.criterion_ce.cuda()
            if criterion in ['Dice', 'Multi']:
                self.criterion_dice = self.criterion_dice.cuda()

            self.criterion_mse = self.criterion_mse.cuda()

    def __define_optimizer(self, learning_rate, weight_decay,
                           lr_drop_factor, lr_drop_patience, optimizer='Adam'):
        assert optimizer in ['RMSprop', 'Adam', 'Adadelta', 'SGD']

        parameters = ifilter(lambda p: p.requires_grad,
                             self.model.parameters())

        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(
                parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(
                parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters, lr=learning_rate, momentum=0.9,
                weight_decay=weight_decay)

        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=lr_drop_factor,
            patience=lr_drop_patience, verbose=True)

    @staticmethod
    def __get_loss_averager():
        return averager()

    def __minibatch(self, train_test_iter, clip_grad_norm,
                    criterion_type, train_cnn=True, mode='training',
                    debug=False):
        assert mode in ['training',
                        'test'], 'Mode must be either "training" or "test"'

        if mode == 'training':
            for param in self.model.parameters():
                param.requires_grad = True
            if not train_cnn:
                for param in self.model.cnn.parameters():
                    param.requires_grad = False
            self.model.train()
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        cpu_images, cpu_sem_seg_annotations, \
            cpu_ins_seg_annotations, cpu_n_objects = train_test_iter.next()
        cpu_images = cpu_images.contiguous()
        cpu_sem_seg_annotations = cpu_sem_seg_annotations.contiguous()
        cpu_ins_seg_annotations = cpu_ins_seg_annotations.contiguous()
        cpu_n_objects = cpu_n_objects.contiguous()

        if self.usegpu:
            gpu_images = cpu_images.cuda(async=True)
            gpu_sem_seg_annotations = cpu_sem_seg_annotations.cuda(async=True)
            gpu_ins_seg_annotations = cpu_ins_seg_annotations.cuda(async=True)
            gpu_n_objects = cpu_n_objects.cuda(async=True)
        else:
            gpu_images = cpu_images
            gpu_sem_seg_annotations = cpu_sem_seg_annotations
            gpu_ins_seg_annotations = cpu_ins_seg_annotations
            gpu_n_objects = cpu_n_objects

        gpu_images, gpu_sem_seg_annotations, \
            gpu_ins_seg_annotations, gpu_n_objects = \
            self.__define_input_variables(gpu_images,
                                          gpu_sem_seg_annotations,
                                          gpu_ins_seg_annotations,
                                          gpu_n_objects, mode)
        gpu_n_objects_normalized = gpu_n_objects.float() / self.max_n_objects

        sem_seg_predictions, ins_seg_predictions, \
            n_objects_predictions = self.model(gpu_images)

        if mode == 'test':
            if debug:
                _vis_prob = np.random.rand()
                if _vis_prob > 0.7:
                    if self.use_instance_segmentation:
                        sem_seg_preds = np.argmax(
                            sem_seg_predictions.data.cpu().numpy(), axis=1)
                        seg_preds = ins_seg_predictions.data.cpu().numpy()

                        _bs, _n_feats = seg_preds.shape[:2]

                        _sample_idx = np.random.randint(_bs)
                        _sem_seg_preds_sample = sem_seg_preds[_sample_idx]
                        _seg_preds_sample = seg_preds[_sample_idx]

                        fg_ins_embeddings = np.stack(
                            [_seg_preds_sample[i][np.where(
                                _sem_seg_preds_sample == 1)]
                                for i in range(_n_feats)], axis=1)
                        _n_fg_samples = fg_ins_embeddings.shape[0]
                        if _n_fg_samples > 0:
                            fg_ins_embeddings = \
                                fg_ins_embeddings[np.random.choice(
                                    range(_n_fg_samples), size=400)]

                            tsne = TSNE(n_components=2, random_state=0)
                            fg_ins_embeddings_vis = tsne.fit_transform(
                                fg_ins_embeddings)

                            if self.instance_seg_vis:
                                self.vis.scatter(X=fg_ins_embeddings_vis,
                                                 win=self.instance_seg_vis,
                                                 opts={'title':
                                                       'Predicted Embeddings \
                                                       for Foreground \
                                                       Predictions',
                                                       'markersize': 2})
                            else:
                                self.instance_seg_vis =\
                                    self.vis.scatter(X=fg_ins_embeddings_vis,
                                                     opts={'title':
                                                           'Predicted \
                                                           Embeddings for \
                                                           Foreground \
                                                           Predictions',
                                                           'markersize': 2})

        cost = 0.0
        out_metrics = dict()

        if self.use_instance_segmentation:
            disc_cost = self.criterion_discriminative(
                ins_seg_predictions, gpu_ins_seg_annotations.float(),
                cpu_n_objects, self.max_n_objects)
            cost += disc_cost
            out_metrics['Discriminative Cost'] = disc_cost.data

        if criterion_type in ['CE', 'Multi']:
            _, gpu_sem_seg_annotations_criterion_ce = \
                gpu_sem_seg_annotations.max(1)
            ce_cost = self.criterion_ce(
                sem_seg_predictions.permute(0, 2, 3, 1).contiguous().view(
                    -1, self.n_classes),
                gpu_sem_seg_annotations_criterion_ce.view(-1))
            cost += ce_cost
            out_metrics['CE Cost'] = ce_cost.data
        if criterion_type in ['Dice', 'Multi']:
            dice_cost = self.criterion_dice(
                sem_seg_predictions, gpu_sem_seg_annotations)
            cost += dice_cost
            out_metrics['Dice Cost'] = dice_cost.data

        mse_cost = self.criterion_mse(
            n_objects_predictions, gpu_n_objects_normalized)
        cost += mse_cost
        out_metrics['MSE Cost'] = mse_cost.data

        if mode == 'training':
            self.model.zero_grad()
            cost.backward()
            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), clip_grad_norm)
            self.optimizer.step()

        return out_metrics

    def __test(self, test_loader, criterion_type, epoch, debug):

        n_minibatches = len(test_loader)

        test_iter = iter(test_loader)

        out_metrics = dict()
        for minibatch_index in range(n_minibatches):
            mb_out_metrics = self.__minibatch(
                test_iter, 0.0, criterion_type, train_cnn=False, mode='test',
                debug=debug)
            for mk, mv in mb_out_metrics.iteritems():
                if mk not in out_metrics:
                    out_metrics[mk] = []
                out_metrics[mk].append(mv)

        test_metric_vis_data, test_metric_vis_legend = [], []
        metrics_as_str = 'Testing:     [METRIC]'
        for mk, mv in out_metrics.iteritems():
            out_metrics[mk] = torch.stack(mv, dim=0).mean()
            metrics_as_str += ' {} : {} |'.format(mk, out_metrics[mk])

            test_metric_vis_data.append(out_metrics[mk])
            test_metric_vis_legend.append(mk)

        print metrics_as_str

        test_metric_vis_data = np.expand_dims(
            np.array(test_metric_vis_data), 0)

        if self.test_metric_vis:
            self.vis.line(X=np.array([epoch]),
                          Y=test_metric_vis_data,
                          win=self.test_metric_vis,
                          update='append')
        else:
            self.test_metric_vis = self.vis.line(X=np.array([epoch]),
                                                 Y=test_metric_vis_data,
                                                 opts={'legend':
                                                       test_metric_vis_legend,
                                                       'title': 'Test Metrics',
                                                       'showlegend': True,
                                                       'xlabel': 'Epoch',
                                                       'ylabel': 'Metric'})

        return out_metrics

    def fit(self, criterion_type, delta_var, delta_dist, norm,
            learning_rate, weight_decay, clip_grad_norm,
            lr_drop_factor, lr_drop_patience, optimize_bg, optimizer,
            train_cnn, n_epochs, class_weights, train_loader, test_loader,
            model_save_path, debug):

        assert criterion_type in ['CE', 'Dice', 'Multi']

        training_log_file = open(os.path.join(
            model_save_path, 'training.log'), 'w')
        validation_log_file = open(os.path.join(
            model_save_path, 'validation.log'), 'w')

        training_log_file.write('Epoch,Cost\n')
        validation_log_file.write('Epoch,Cost\n')

        self.__define_criterion(class_weights, delta_var, delta_dist,
                                norm=norm, optimize_bg=optimize_bg,
                                criterion=criterion_type)
        self.__define_optimizer(learning_rate, weight_decay,
                                lr_drop_factor, lr_drop_patience,
                                optimizer=optimizer)

        self.__test(test_loader, criterion_type, -1.0, debug)

        best_val_cost = np.Inf
        for epoch in range(n_epochs):
            epoch_start = time.time()

            train_iter = iter(train_loader)
            n_minibatches = len(train_loader)

            train_out_metrics = dict()

            minibatch_index = 0
            while minibatch_index < n_minibatches:
                mb_out_metrics = self.__minibatch(train_iter, clip_grad_norm,
                                                  criterion_type,
                                                  train_cnn=train_cnn,
                                                  mode='training', debug=debug)
                for mk, mv in mb_out_metrics.iteritems():
                    if mk not in train_out_metrics:
                        train_out_metrics[mk] = []
                    train_out_metrics[mk].append(mv)

                minibatch_index += 1

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start

            training_metric_vis_data, training_metric_vis_legend = [], []

            print 'Epoch : [{}/{}] - [{}]'.format(epoch,
                                                  n_epochs, epoch_duration)
            metrics_as_str = 'Training:    [METRIC]'
            for mk, mv in train_out_metrics.iteritems():
                train_out_metrics[mk] = torch.stack(mv, dim=0).mean()
                metrics_as_str += ' {} : {} |'.format(mk,
                                                      train_out_metrics[mk])

                training_metric_vis_data.append(train_out_metrics[mk])
                training_metric_vis_legend.append(mk)

            print metrics_as_str

            training_metric_vis_data = np.expand_dims(
                np.array(training_metric_vis_data), 0)

            if self.training_metric_vis:
                self.vis.line(X=np.array([epoch]),
                              Y=training_metric_vis_data,
                              win=self.training_metric_vis, update='append')
            else:
                self.training_metric_vis = self.vis.line(
                    X=np.array([epoch]), Y=training_metric_vis_data,
                    opts={'legend': training_metric_vis_legend,
                          'title': 'Training Metrics',
                          'showlegend': True, 'xlabel': 'Epoch',
                          'ylabel': 'Metric'})

            val_out_metrics = self.__test(
                test_loader, criterion_type, epoch, debug)
            if self.use_instance_segmentation:
                val_cost = val_out_metrics['Discriminative Cost']
                train_cost = train_out_metrics['Discriminative Cost']
            elif criterion_type in ['Dice', 'Multi']:
                val_cost = val_out_metrics['Dice Cost']
                train_cost = train_out_metrics['Dice Cost']
            else:
                val_cost = val_out_metrics['CE Cost']
                train_cost = train_out_metrics['CE Cost']

            self.lr_scheduler.step(val_cost)

            is_best_model = val_cost <= best_val_cost

            if is_best_model:
                best_val_cost = val_cost
                torch.save(self.model.state_dict(), os.path.join(
                    model_save_path, 'model_{}_{}.pth'.format(epoch,
                                                              val_cost)))

            training_log_file.write('{},{}\n'.format(epoch, train_cost))
            validation_log_file.write('{},{}\n'.format(epoch, val_cost))
            training_log_file.flush()
            validation_log_file.flush()

        training_log_file.close()
        validation_log_file.close()

    def predict(self, images):

        assert len(images.size()) == 4  # b, c, h, w

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        images = images.contiguous()
        if self.usegpu:
            images = images.cuda(async=True)

        images = self.__define_variable(images, volatile=True)

        sem_seg_predictions, ins_seg_predictions, n_objects_predictions = \
            self.model(images)

        sem_seg_predictions = torch.nn.functional.softmax(
            sem_seg_predictions, dim=1)

        n_objects_predictions = n_objects_predictions * self.max_n_objects
        n_objects_predictions = torch.round(n_objects_predictions).int()

        sem_seg_predictions = sem_seg_predictions.data.cpu()
        ins_seg_predictions = ins_seg_predictions.data.cpu()
        n_objects_predictions = n_objects_predictions.data.cpu()

        return sem_seg_predictions, ins_seg_predictions, n_objects_predictions


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`."""

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
