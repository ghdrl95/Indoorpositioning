'''
베이시안 옵티마이저 적용해 RNN기법으로 위치 측위
'''



import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from GPyOpt.methods import BayesianOptimization
#하이퍼파라미터 정의
cross_validation = 5
input_size = 620
output_size = 2
batch_size = 50
path = 5
#베이시안 옵티마이저 파라미터. EPOCH, DROPOUT, CELLSIZE,
domain = [{'name': 'lr_rate',
          'type': 'continuous',
          'domain': (0.0001, 0.05)},
          {'name': 'dropout',
          'type': 'continuous',
          'domain': (0.5, 1),
          'dimensionality': 1},
          {'name': 'LSTMlength',
           'type': 'discrete',
           'domain':(2, 3, 4, 5, 6, 7),
           'dimensionality': 1},
           {'name': 'hiddencell',
           'type': 'continuous',
           'domain':(100, input_size*2)},
          {'name': 'epoch',
           'type': 'continuous',
           'domain':(20, 1000),
           'dimensionality': 1}]
#데이터 로드
data_train = np.loadtxt('.\\data\\output_%s_train.csv'%path,delimiter=',')
data_test = np.loadtxt('.\\data\\output_%s_test.csv'%path,delimiter=',')


#셔플
np.random.shuffle(data_train)
np.random.shuffle(data_test)

#data_sample = []
#each_sample_num = data.shape[0] // cross_validation
#for i in range(cross_validation):
#    data_sample.append(data[i*each_sample_num : (i+1)*each_sample_num])

minimum_error = 99999
minimum_snapshot = []

#크로스 벨리데이션에서 사용할 그래프 생성
def set_graph(arg):
    #global data_sample
    global data
    global input_size,output_size,cross_validation,path,each_sample_num
    #데이터 가져오기
    #learning_rate = arg[0]
    #dropout = arg[1]
    #lstmlength = int(arg[2])
    #hiddencell_num = int(arg[3])
    #epoch = int(arg[4])
    learning_rate = arg[0][0]
    dropout = arg[0][1]
    lstmlength = int(arg[0][2])
    hiddencell_num = int(arg[0][3])
    epoch = int(arg[0][4])
    #learning_rate = 0.01
    #dropout = 0.65
    #lstmlength = 2
    #hiddencell_num = 1884
    #epoch = 20
    ## 그래프 생성
    # 입력
    tf.reset_default_graph()
    input = tf.placeholder(tf.float32, shape=[None, input_size * path], name='input')
    in_reshape = tf.reshape(input,[-1,path,input_size ])
    output = tf.placeholder(tf.float32, shape=[None, output_size], name='output')
    in_dropout = tf.placeholder(tf.float32)

    # 연산
    lstms=[]
    for i in range(lstmlength):
        lstm_cell =tf.contrib.rnn.LayerNormBasicLSTMCell(hiddencell_num,dropout_keep_prob=in_dropout)
        lstms.append(lstm_cell)
    cell = tf.contrib.rnn.MultiRNNCell(lstms)
    lstm_outputs,lstm_state = tf.nn.dynamic_rnn(cell, in_reshape, dtype=tf.float32)

    rnn_out = lstm_outputs[:,-1]



    # 출력
    model_output = slim.fully_connected(rnn_out, output_size, None)
    loss = tf.sqrt(tf.reduce_sum(tf.square(output-model_output), axis=1))
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #크로스 벨리데이션 사용
    #오차값 평균 저장
    error_avg = 0

    test_data_x = data_test[:, :-2]
    test_data_y = data_test[:, -2:]
    train_data_x = data_train[:, :-2]
    train_data_y = data_train[:, -2 :]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #학습
        for i in range(epoch):

            for batch_idx in range(train_data_x.shape[0] // batch_size):
                _ = sess.run(optimizer, feed_dict={input : train_data_x[batch_idx*batch_size : (batch_idx+1)*batch_size],
                                                   output: train_data_y[batch_idx*batch_size : (batch_idx+1)*batch_size],
                                                   in_dropout:dropout})


        #테스트
        loss = 0
        test_batch_size = test_data_x.shape[0] // batch_size
        for batch_idx in range(test_batch_size):
            loss += sess.run(cost, feed_dict={input : test_data_x[batch_idx*batch_size : (batch_idx+1)*batch_size],
                                              output: test_data_y[batch_idx*batch_size : (batch_idx+1)*batch_size],
                                              in_dropout:1.0})
        error_avg += (loss / test_batch_size)
        print("error 확인 :",loss / test_batch_size)

    print("최종 에러 : ", error_avg)
    global minimum_error
    if error_avg < minimum_error:
        #모델 저장 및 최소값 저장
        minimum_error = error_avg
        minimum_snapshot = [learning_rate, dropout, lstmlength, hiddencell_num, epoch, error_avg]
        np.savetxt("minimum_snapshot.txt", minimum_snapshot,delimiter=',')
        print("minimum :", error_avg, "    snapshot :",minimum_snapshot)

    return error_avg

my_opt = BayesianOptimization(f=set_graph, domain=domain,initial_design_numdata=5)
my_opt.run_optimization(max_iter=10)
print("결과 : ", minimum_error, minimum_snapshot)









