
__all__ = ['AVG']

class AVG:
    def __init__(self,args):
        self.current=None
        self.args = args

    def is_new_best(self,val_log):
        if self.current == None:
            self.current = val_log
            print('new best score = {}, detail{}'.format((val_log['iou']-val_log['pixel_error']-val_log['rand_error'])/3,self.current ))
            return True
        else:
            new_score = (val_log['iou']-val_log['pixel_error']-val_log['rand_error'])/3
            old_score = (self.current['iou']-self.current['pixel_error']-self.current['rand_error'])/3
            if new_score>old_score:
                self.current = val_log
                print('new best score = {}, detail{}'.format((val_log['iou']-val_log['pixel_error']-val_log['rand_error'])/3,self.current ))
                return True
            else:
                return False

