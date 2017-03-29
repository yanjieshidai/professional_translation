
原文链接：<a href="https://blog.altoros.com/using-k-means-clustering-in-tensorflow.html">Using k-means Clustering in TensorFlow - Blog on All Things Cloud Foundry</a>


# 导读

这篇TensorFlow介绍的目标是使用k-means算法对数据进行分组。k-means算法是一个无监督算法，训练集数据不需要标签，数据本身会进行聚类。

# 运行TensorFlow


首先，构造一些均匀分布随机点，并通过tf.constant函数进行构造。然后，从数据集中随机选择初始化中心点。

```python
points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
```


下一步，我们想要对这些二维矩阵做减法。因为这些张量有不同的形状，我们需要扩展这些点到3维。这样，我们才能使用<a href="https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html">Broadcasting feature</a>进行减法操作。

```python
points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)
```


然后，计算各个点到中心点的平均距离，并得出指派集。
```python
distances = tf.reduce_sum(tf.square(tf.sub(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)
```

下一步，我们可以比较每个聚类和指派集，得到每个聚类的掉，并计算均值。这些均值重新定义了聚类中心，并用新的值更新这些中心点。

```python
means = []
for c in xrange(clusters_n):
    means.append(tf.reduce_mean(
      tf.gather(points, 
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, c)
                  ),[1,-1])
               ),reduction_indices=[1]))

new_centroids = tf.concat(0, means)
update_centroids = tf.assign(centroids, new_centroids)
```

这个时候就可以运行这个图了。每次迭代，都会更新中心点，然后返回他们的值，和assignments values一起。

```python
with tf.Session() as sess:
  sess.run(init)
  for step in xrange(iteration_n):
    [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])
```

最后，在坐标系中绘制最终中心点，并使用multi-scatter绘制各个聚类的结果.

```python
print "centroids" + "\n", centroid_values

plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()
```


![](http://pic.jiehouse.com/2017_03_29/using-k-means-clustering-in-tensorflow.png)

