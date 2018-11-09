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

        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        accs_validation = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        feed_dict = {self.model.x: self.data.validation_data,
                     self.model.y: self.data.validation_labels,
                     self.model.is_training: True}
        acc_validation = self.sess.run(self.model.accuracy, feed_dict=feed_dict)
        accs_validation.append(acc_validation)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'acc_validation': acc_validation,
        }
        print('current acc = %010.8f, acc_validation = %010.8f' %(acc, acc_validation))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def test(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False}
        prediction, logits = self.sess.run([self.model.prediction, self.model.logits], feed_dict=feed_dict)
        for k in range(batch_y.shape[0]):
            print(logits[k])
            image_data = batch_x[k].reshape(28,-1)
            plt.imshow(image_data, cmap=plt.cm.gray)
            plt.xlabel("y = %d, predict = %d" %(batch_y[k], prediction[k]))
            plt.title("index = %d" %(k))
            plt.show()
