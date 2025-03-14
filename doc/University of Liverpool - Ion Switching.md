# University of Liverpool - Ion Switching

**Ion Channel Openings Prediction** -

Predicting open ion channels from bioelectric signals with significant volatility and diverse response patterns. Designed and implemented a hybrid GRU-Transformer neural network integrated with RFC Prob features, leveraging category-specific denoising and dynamic learning rate scheduling to enhance accuracy, generalization, and robustness.

This competition predicts the number of open ion channels based on electrophysiological signals from human cells. ***open_channels*** has a discrete probability distribution, with values ranging from **0 to 10 occuring in the training data**. The proability generally seems to decrease exponentially with increase in *open_channels*.

![image.png](University%20of%20Liverpool%20-%20Ion%20Switching%201ae71bdab3cf805794affa2f7d93b073/image.png)

![image.png](University%20of%20Liverpool%20-%20Ion%20Switching%201ae71bdab3cf805794affa2f7d93b073/image%201.png)

data processing，augmentation & feature engineering：

1. Identified underlying patterns correlated with open channel counts and applied normalization techniques to produce cleaner, more consistent data.
2. only lag features【-3,3】& sig*sig
3. random forest classification probability as feature

**Wavenet with SHIFTED-RFC Proba** might lead to leak，but i use https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645

**Implements a grouped cross-validation pipeline using neural networks and Random Forest probability features to predict open ion channels from electrophysiological signals, incorporating advanced training strategies including data augmentation, dynamic learning rate scheduling, and optimized class weighting to enhance model performance and generalization.**

1. Feature engineering
    1. RandomForestClassification window (-20,20) predict the current.
2. Models ensemble + max f1 score+GKF+OOF as target
    1. CBR
    2. GRU
    3. Transformer
    4. WaveNet

```python
def CnnGRUTransformerWave(shape_):
    def cbr(x, out_layer, kernel, stride, dilation):
      x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
      x = BatchNormalization()(x)
      x = Activation("relu")(x)
      return x

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x

    inp = Input(shape = (shape_))
    inp1 = K.reshape(inp, (-1, shape_[0]//8, shape_[1]))
    x = cbr(inp1, 64, 1, 1, 1)
    x1 = MaxPooling1D(data_format='channels_first')(x)
    x2 = AveragePooling1D(data_format='channels_first')(x)
    x = Concatenate()([x1,x2])
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x, slf_attn = MultiHeadAttention(n_head=4, d_model=128, d_k=64, d_v=64, dropout=0.1)(x, x, x)
    x1 = MaxPooling1D(data_format='channels_first')(x)
    x2 = AveragePooling1D(data_format='channels_first')(x)
    x = Concatenate()([x1,x2])
    x = Dropout(0.2)(x)
    out1 = Dense(11, activation = 'softmax', name = 'out1')(x)
    out1 = K.reshape(out1, (-1, shape_[0], 11)) 
    x = Concatenate()([inp, out1])
    
    x = cbr(x, 64, 7, 1, 1)
    x = BatchNormalization()(x)
    x = wave_block(x, 16, 3, 12)
    x = BatchNormalization()(x)
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
    x = wave_block(x, 128, 3, 1)
    x = cbr(x, 32, 7, 1, 1)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(11, activation = 'relu', name = 'out')(x)
    out = K.reshape(out, (-1, shape_[0], 11))
    out = Add()([out1, out])
    out = Activation('softmax')(out)
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model
```

1. reduce_mem_usage

This function efficiently reduces pandas DataFrame memory usage by dynamically optimizing data types.

This function reduces pandas DataFrame memory usage by dynamically optimizing numeric column data types, converting integers and floats to the smallest possible types based on their value ranges, significantly decreasing memory consumption without losing data precision.

```python
def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df
```

loss：`CategoricalCrossentropy`

metric: accuracy

best choose：f1 score.

Not working：

1. flip， jitter，rotate data
2. give sample weight