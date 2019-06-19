# 保存したnpyファイルが読み込め、正常であるかを確認する
import numpy

npy_names = ['param_q2_opt_0.npy',
             'param_q2_opt_1E-1.npy',
             'param_q2_opt_1E-2.npy',
             'param_q2_opt_1E-3.npy',
             'param_q2_opt_1E-4.npy',
             'param_q2_opt_1E-5.npy',
             'param_q2_opt_1E-6.npy',
             'param_q2_opt_1E-7.npy',
             'param_q2_opt_1E-8.npy',
             'param_q2_opt_1.npy',
             'param_q2_opt_5E-1.npy',
             'param_q2_opt_5E-2.npy',
             'param_q2_opt_5E-3.npy',
             'param_q2_opt_5E-4.npy',
             'param_q2_opt_5E-5.npy',
             'param_q2_opt_5E-6.npy',
             'param_q2_opt_5E-7.npy',
             'param_q2_opt_5E-8.npy']

for name in npy_names:
    npy = numpy.load(name)
    print(*[numpy.round(d, 1) for d in npy])
