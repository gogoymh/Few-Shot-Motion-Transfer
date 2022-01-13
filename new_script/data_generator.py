import numpy as np
import matplotlib.pyplot as plt


def generate(number=100, seed=None, plot=True, num_class=3,sigma=1.0):
    
    class_number = (number // num_class)
   
    print("%d data points generated." % (number), end=" ")
    
    if seed is not None:
        np.random.seed(seed)
        print('Seed is %d.' % seed)
    else:
        print('Seed is random.')
    
    sigma2=sigma**2
    mean_dict={0:[2,0],1:[2,2],2:[0,2],3:[0,0]}
    cov_dict={0:[[0.25,-0.1],[-0.1,0.15]],1:[[0.2,0],[0,0.25]],2:[[0.25,-0.15],[-0.15,0.4]],3:[[0.4,0],[0,0.4]]}
    color_dict={0:'red',1:'blue',2:'green',3:'black'}
    marker_dict={0:"o",1:"x",2:"^",3:"s"}

    input_value=None
    output_value=None
    
    for i in range(num_class):
        x_data=np.random.multivariate_normal(mean_dict[i], np.array(cov_dict[i])*sigma2, class_number).T + np.random.uniform(-0.3,0.3, class_number)
        
        if input_value is None:
          input_value = x_data
        else:
          input_value = np.concatenate((x_data,input_value),axis=1)

        labels = (i*np.ones((1,class_number))).astype('uint8')

        if output_value is None:
          output_value = labels
        else:
          output_value=np.concatenate((labels,output_value),axis=1)

        if plot:
            plt.scatter(x_data[0], x_data[1], color=color_dict[i], marker=marker_dict[i], s=10)

    if plot:
        plt.axvline(x=0, ymin=-2, ymax=5, color='black', linestyle='--', linewidth=0.5)
        plt.axhline(y=0, xmin=-2, xmax=5, color='black', linestyle='--', linewidth=0.5)
        plt.axis([-2,5,-2,5])
        plt.show()
        plt.close()


    output_value = output_value.reshape((-1,))

    return input_value.T, output_value.T

if __name__ == '__main__':
   x, y = generate(1000, 0, True, True,3,1.0)



