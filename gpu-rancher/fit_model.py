from keras import optimizers
from keras.models import Sequential, Model

def train_model_k_folds(model, train_data, train_label, model_out, model_init_weights, epochs, kfolds):
    batch_size = 32
    kf = model_selection.KFold(n_splits=kfolds, shuffle=True)
    score_func = metrics.roc_auc_score

    i = 0
    models_stats = {}
    for train_ixs, valid_ixs in kf.split(train_data):
        x_train = train_data[train_ixs]
        x_valid = train_data[valid_ixs]
        y_train = train_label[train_ixs]
        y_valid = train_label[valid_ixs]

        gen = ImageDataGenerator(
            rotation_range = 30,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        
        #re-initialzie the weights of the model on each run
        model.load_weights(model_init_weights)
        model_out_file = '/scratch/yns207/data_invasive/{}_{}.model'.format(model_out, str(i))
        model_checkpoint = ModelCheckpoint(model_out_file, 
                                            monitor='val_loss', 
                                            save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto')

        hist = model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=(len(x_train)//batch_size)+1,
                            validation_data=(x_valid,y_valid),
                            validation_steps=(len(x_valid)//batch_size)+1,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[early_stopping, model_checkpoint])
        
        model.load_weights(model_out_file)
        
        eval_tr = model.evaluate(x_train, y_train)
        eval_va = model.evaluate(x_valid, y_valid)
        
        tr_score = score_func(y_train, model.predict(x_train)[:, 0])
        va_score = score_func(y_valid, model.predict(x_valid)[:, 0])
        
        print('\n')
        print('kfold: {}'.format(str(i)))
        print('best model train acc: {}, loss: {}'.format(eval_tr[1], eval_tr[0]))
        print('best model valid acc: {}, loss: {}'.format(eval_va[1], eval_va[0]))
        print('best model train aroc score: {}, valid aroc score: {}'.format(tr_score, va_score))
        print('\n')
        models_stats[model_out_file] = {'score_tr_va':[tr_score, va_score], 'train_acc_loss':[eval_tr[1], eval_tr[0]], 'val_acc_loss':[eval_va[1], eval_va[0]]}
        i += 1
        
    return models_stats



def parse_args(args):
    currentdate = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', dest='inputs', required=True, nargs='+', help='These inputs are paths to your bson files. Required.')
    parser.add_argument('-l', '--log', dest='log', default=expanduser('/scratch/yns207/job_logs/fit_model_'+currentdate+'.log'), help='This is the path to where your output log should be. Required')
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    logging.basicConfig(filename=args.log, level=logging.INFO)
    if args.uniquefield:
        merge_bson_unique(args.output, args.inputs, args.uniquefield)
    else:
        merge_bson(args.output, args.inputs)