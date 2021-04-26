import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Layer
import tensorflow.keras.backend as K

##point cloud operations
def pointcloud_distance(point_cloud):
    #'''Compute angular distances between each point and all others in a point cloud.
    #    point_cloud: tensor (batch_size,npoints,dims)
    #returns:
    #    distance tensor: tensor (batch_size,npoints,npoints)
    #'''
    assert len(K.int_shape(point_cloud)) == 3
    # batchsize = point_cloud.get_shape().as_list()[0]

    #Calculate the L2 norm between pointcloud and itself transposed
    #||x1 - x0||2 = sqrt( ||x0||^2 + ||x1||^2 - 2 * x1.x0 )
    #drop the squareroot, won't change relative distances
    
    point_cloud_transpose = K.permute_dimensions(point_cloud, (0,2,1))
    point_cloudx0square = K.sum(K.square(point_cloud),axis=-1,keepdims=True)

    point_cloudx1square = K.permute_dimensions(point_cloudx0square,(0,2,1)) ## same as x0 square but transposed
    point_cloudx1x0 = K.batch_dot(point_cloud,point_cloud_transpose) # tf.matmul(transpose_b=True) <- check if that is [0,2,1] or [2,1,0]
    point_cloudx1x0 *= -2

    return point_cloudx1square + point_cloudx0square + point_cloudx1x0

def knn(distance,k):
    #'''Get indices of the k Nearest Neighbours using distance tensor.
    #distance: (batch_size,npoints,npoints)
    #returns: (batch_size,npoints,k)'''

    #top_k returns k largest values, we want k smallest -> negative distance
    _,indx = tf.nn.top_k(-distance,k=k)

    return indx

def edgefeatures(point_cloud,nn_index):
    #'''Get the edge features from the point cloud and k-neighbours. Edges defined as displacement of neighbour from point.
    #    point_cloud: tensor (batch_size,npoints,dims)
    #    nn_index: tensor (batch_size,npoints,k)
    #returns: tensor (batch_size,npoints,k+1,ndim)
    #'''
    pc_shape = K.shape(point_cloud)

    # pc_shape = point_cloud.get_shape().as_list()
    batch_size = pc_shape[0]
    npoints,ndim = K.int_shape(point_cloud)[1:3]#pc_shape[1]
    # ndim = K.int_shape(pc_shape)[2]
    # k = nn_index.get_shape().as_list()[2]
    k = K.int_shape(nn_index)[-1]

    pc_centre = point_cloud

    pc_start_idx = K.arange(0,batch_size) * npoints # create starting point of each cloud when flattening into (batchsize,dims_for_allpoints_sequentially)
    pc_start_idx = K.reshape(pc_start_idx,[batch_size,1,1])
    # pc_start_idx = K.expand_dims((K.expand_dims(pc_start_idx)))

    # flatten point cloud for gather function, done along axis=0 -> batchsize, gets features based on index
    pc_flat = K.reshape(point_cloud,[-1,ndim])
    # get the features of the neighbouring points, (batchsize,npoints,len(nn_index)=k,ndim)
    pc_neighbours = K.gather(pc_flat,nn_index+pc_start_idx)

    #now create tensor of shape (batchsize,npoint,k,ndim) and fill it with pc_centre - pc_neighbour
    pc_centre = K.expand_dims(pc_centre,axis=-2)
    pc_centres = K.tile(pc_centre,[1,1,k,1])

    # return edges (distance from neighbour to centre) and central value (self connecting edge)
    edges = K.concatenate([pc_centre,pc_neighbours - pc_centres],axis=2)
    return edges



class EdgeConv(Layer):
    def __init__(self,k,
                filters,
                point_cloud_dimensions=2,
                dynamic_coordinates=True,
                shortcut_input_features=True,
                pooling='Max',
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        assert pooling in ['Max', 'Mean', 'Sum']

        
        self.k = k
        self.filters = filters
        self.point_cloud_dimensions = point_cloud_dimensions
        self.dynamic_coordinates = dynamic_coordinates
        self.shortcut_input_features = shortcut_input_features
        self.pooling = pooling
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.supports_masking = True

        self.conv = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            data_format = data_format,
            dilation_rate = dilation_rate,
            activation = activation,
            use_bias = use_bias,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer,
            kernel_constraint = kernel_constraint,
            bias_constraint = bias_constraint,
            **kwargs)
        super(EdgeConv,self).__init__()

    def get_config(self):
        config = {
            'k':self.k,
            'filters':self.filters,
            'point_cloud_dimensions':self.point_cloud_dimensions,
            'dynamic_coordinates':self.dynamic_coordinates,
            'pooling':self.pooling,
            'kernel_size':self.kernel_size,
            'strides':self.strides,
            'padding':self.padding,
            'data_format':self.data_format,
            'dilation_rate':self.dilation_rate,
            'activation':self.activation,
            'use_bias':self.use_bias,
            'kernel_initializer':self.kernel_initializer,
            'bias_initializer':self.bias_initializer,
            'kernel_regularizer':self.kernel_regularizer,
            'bias_regularizer':self.bias_regularizer,
            'activity_regularizer':self.activity_regularizer,
            'kernel_constraint':self.kernel_constraint,
            'bias_constraint':self.bias_constraint
        }
            
        base_config = super(EdgeConv,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def call(self, inputs, mask=None):

        #Get the angular coordinates
        points = inputs[:,:,:self.point_cloud_dimensions]
        #Do we want dynamic spatial coordinates (ie dynamic graph, if yes then spatial coordinates must enter convolution)
        features = inputs
        if self.dynamic_coordinates is not True:
            features = features[:,:,self.point_cloud_dimensions:]

        #If masking, move masked points far from any other
        if mask is not None:
            points = points + 99. * K.expand_dims(K.cast(mask,dtype='float32'),-1)

        #Get edge features
        dist_mat = pointcloud_distance(points)
        kneighbours = knn(dist_mat,self.k)

        edges = edgefeatures(features,kneighbours)

        #Perform (1,1) 2D convolution on edge features
        outputs = self.conv(edges)

        #Apply pooling operation on k-neighbours
        if self.pooling=='Max':
            outputs = K.max(outputs,axis=-2)
        elif self.pooling=='Mean':
            outputs = K.mean(outputs,axis=-2)
        elif self.pooling=='Sum':
            outputs = K.sum(outputs,axis=-2)

        #set all feature values to zero if masked
        if mask is not None:
            outputs = outputs * (1 - K.expand_dims(K.cast(mask,dtype='float32'),-1))

        #If retaining coordinates add them back to beginning of tensor
        if self.dynamic_coordinates is not True:
            outputs = K.concatenate([points,outputs],axis=-1)

        #If concatenating old features back into feature tensor
        if self.shortcut_input_features is True:
            outputs = K.concatenate([outputs,features],axis=-1)
 
        return outputs




