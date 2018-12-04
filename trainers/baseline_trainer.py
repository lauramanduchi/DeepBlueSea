from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import datetime


class BaselineTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(BaselineTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        dev_losses = []
        dev_accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
            cur_it = self.model.global_step_tensor.eval(self.sess)
            if cur_it % self.config.evaluate_every == 0:
                print("\nEvaluation:")
                loss, acc = self.dev_step()
                dev_losses.append(loss)
                dev_accs.append(acc)
                print("")

        loss = np.mean(losses)
        acc = np.mean(accs)
        dev_loss = np.mean(dev_losses)
        dev_acc = np.mean(dev_accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        print("SUMMARY: epoch {}, loss {:g}, acc {:g}, dev_loss {:g}, dev_acc {:g}".format(cur_epoch, loss, acc, dev_loss, dev_acc))
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        step = self.model.global_step_tensor.eval(self.sess)
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))

        return loss, acc

    def dev_step(self):
        batch_x, batch_y = next(self.data.next_batch_dev(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False}

        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        step = self.model.global_step_tensor.eval(self.sess)
        print("VAL {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))

        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }

        self.logger.summarize(step, summarizer="dev", summaries_dict=summaries_dict)
        return loss, acc
