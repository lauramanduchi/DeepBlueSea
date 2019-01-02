import numpy as np
from tqdm import tqdm
from tqdm import trange

from base.base_train import BaseTrain


class FasterRcnnTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(FasterRcnnTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = trange(self.config.num_iter_per_epoch, desc='Training;', leave=True)
        losses = []
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

        loss = np.mean(losses)
        print('Epoch Loss: {}'.format(loss))

        summaries_dict = {
            'loss': loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        feed_dict = {self.model.handle: self.sess.run(self.data.dataset_iterator.string_handle())}
        _, loss, summaries = self.sess.run([self.model.train_step, self.model.mse, self.model.summaries],
                                           feed_dict=feed_dict)

        return loss, summaries
