# GPGPU-Experiment
GPGPUの実験

## 概要
ほとんどのコンピューターにはGPUという映像処理を専門に行う処理機構が搭載されている。その中でも、GPUのみで構成されるハードウェアをグラフィックスボードと呼ぶ。
ディスプレイはピクセルの集合によって構成されており、これを高速に切り替えることでユーザーは映像を知覚している。
近年のディスプレイは1ドットを16bitまたは24bit深度で表現しており、フルHDの映像を出力するにはおよそ12MBや24MBのデータが必要になる。
### 旧来の映像出力方法
GPUという概念が確立されていなかった頃は、メインメモリの一部をグラフィックスメモリとして利用してCPUによって描画を行っていた。しかし、そのメモリ量はディスプレイに表示する全てのデータ量には満たないため、RGBのいずれかの色情報のみが記録できる領域のみを確保し、色情報を切り替えながら処理を行っていた。
今日でも、一部のコンピュータにおいてもこの実装を採用しているものがある。
### GPUの特徴とCUDA
グラフィックス処理は一度に大量のデータを処理する必要があるため、簡単な処理のみが可能なプロセッサが大量に搭載されている。そのため、全体としての処理は行列計算となる。そのため、大規模な並列演算が可能となる。
NVIDIAは、映像処理の他に行列計算などをグラフィックスボード上で行うことができるようにライブラリを作成した。それがCUDAである。
```math
A = \left(
    \begin{array}{ccc}
      a_{0,0} & \ldots &  \\
      \vdots & \ddots & \vdots \\
       & \ldots & a_{n,n}
    \end{array}
  \right)
```
```math
B = \left(
    \begin{array}{ccc}
      b_{0,0} & \ldots &  \\
      \vdots & \ddots & \vdots \\
       & \ldots & b_{n,n}
    \end{array}
  \right)
```
```math
A+B
```
などの行列演算が可能である。