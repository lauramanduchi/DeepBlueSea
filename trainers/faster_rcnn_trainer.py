import numpy as np
from tqdm import trange
import datetime

from base.base_train import BaseTrain


class FasterRcnnTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(FasterRcnnTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = trange(self.config.num_iter_per_epoch, desc='Training;', leave=True)
        losses = []
        dev_losses = []
        for _ in loop:
            loss, summaries = self.train_step()
            losses.append(loss)
            loop.set_description("Training; " + 'Current Batch Loss: {}'.format(loss))
            loop.refresh()

            # Added a summariser for the images on each batch. You can declare what you want summarised
            # within the model (see: faster_rcnn_model.py,
            # eg. tf.summary.image(name = 'input_images', tensor=self.x, max_outputs=3))
            cur_it = self.model.global_step_tensor.eval(self.sess)
            self.logger.train_summary_writer.add_summary(summaries, global_step=cur_it)

            if cur_it % self.config.evaluate_every == 0:
                print("\nEvaluation:")
                loss, acc = self.dev_step()
                dev_losses.append(loss)
                print("")

        loss = np.mean(losses)
        dev_loss = np.mean(dev_losses)
        print('SUMMARY: Epoch Loss: {}, Dev Loss: {}'.format(loss, dev_loss))

        summaries_dict = {
            'loss': loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)


    # def train_step(self):
    #     feed_dict = {self.model.handle: self.sess.run(self.data.dataset_iterator.string_handle())}
    #     _, loss, summaries = self.sess.run([self.model.train_step, self.model.loss, self.model.summaries],
    #                                        feed_dict=feed_dict)
    #
    #     return loss, summaries

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y_map: batch_y, self.model.is_training: True}
        _, loss, summaries = self.sess.run([self.model.train_step, self.model.loss, self.model.summaries],
                                     feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        step = self.model.global_step_tensor.eval(self.sess)
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

        return loss, summaries

    def dev_step(self):
        batch_x, batch_y = next(self.data.next_batch_dev(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y_map: batch_y, self.model.is_training: False}

        _, loss, summaries = self.sess.run([self.model.train_step, self.model.loss, self.model.summaries],
                                           feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        step = self.model.global_step_tensor.eval(self.sess)
        print("VAL {}: step {}, loss {:g}".format(time_str, step, loss))

        summaries_dict = {
            'loss': loss
        }

        self.logger.summarize(step, summarizer="dev", summaries_dict=summaries_dict)

        return loss, summaries