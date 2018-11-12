from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        cur_epoch     = self.model.cur_epoch_tensor  .eval(self.sess)
        cur_it        = self.model.global_step_tensor.eval(self.sess)
        learning_rate = self.sess.run(self.model.learning_rate)
        print('Training epoch %04d/%04d, learning_rate = %010.8f' %(cur_epoch,self.config.num_epochs,learning_rate))

        loop = tqdm(range(self.config.num_iter_per_epoch), desc='train')
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        train_loss = np.mean(losses)
        train_acc = np.mean(accs)

        loop = tqdm(range(int(0.5+self.data.valid_size/self.config.batch_size)), desc='valid')
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.valid_step()
            losses.append(loss)
            accs.append(acc)
        valid_loss = np.mean(losses)
        valid_acc  = np.mean(accs)

        summaries_dict = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }
        print('current train_acc = %010.8f, valid_acc = %010.8f' %(train_acc, valid_acc))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.train_next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def valid_step(self):
        batch_x, batch_y = next(self.data.valid_next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy], feed_dict=feed_dict)
        return loss, acc

    def test_step(self):
        batch_x, batch_y = next(self.data.test_next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy], feed_dict=feed_dict)
        return loss, acc

    def test(self):
        loop = tqdm(range(int(0.5+self.data.test_size/self.config.batch_size)), desc='test auto statistic')
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.test_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc  = np.mean(accs)
        print("test_acc =", acc, "test_loss =", loss)
        
        loop = tqdm(range(self.data.test_size), desc='test manual')
        for k in loop:
            x = self.data.test_data[[k]]
            y = self.data.test_labels[[k]]
            feed_dict = {self.model.x: x, self.model.y: y, self.model.is_training: False}
            prediction, logits = self.sess.run([self.model.prediction, self.model.logits], feed_dict=feed_dict)
            if prediction[0] != y[0]:
                print(logits)
                image_data = x[0].reshape(28,-1)
                plt.imshow(image_data, cmap=plt.cm.gray)
                plt.xlabel("y = %d, predict = %d" %(y[0], prediction[0]))
                plt.title("index = %d" %(k))
                plt.show(block=True)
