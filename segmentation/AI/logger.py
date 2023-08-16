from datetime import datetime

class Logger():
    def __init__(self, log_file=None, overwrite=True, track_metric=True, metric='aP') -> None:
        self.log_file = log_file
        self.track_metric = track_metric
        self.metric = metric
        self.best_metric = -1
        self.history = []
        self.starting_time = datetime.now()
        self.best_epoch = 0

        if log_file:
            if overwrite:
                open(log_file, 'w').close()
        
    def log(self, log_dict):
        time_past = f"{str((datetime.now() - self.starting_time)).split('.')[0]}".replace(':', ';')
        if self.track_metric and log_dict[self.metric] > self.best_metric:
            self.best_metric = log_dict[self.metric]
            self.best_epoch = int(log_dict['epoch'])
            
        if self.log_file:
            with open(self.log_file, 'a') as f:
                line = 'HMS: {}, '.format(time_past) + ', '.join([': '.join([k, str(log_dict[k])]) for k in [k for k in log_dict.keys() if not hasattr(log_dict[k], 'keys')]])
                for uk in [k for k in log_dict.keys() if hasattr(log_dict[k], 'keys')]:
                    line += ', ' + ', '.join([': '.join(['{}_{}'.format(uk, lk), str(log_dict[uk][lk])]) for lk in log_dict[uk].keys()])
                line += '\n'
                f.write(line)

        return self.best_epoch
        