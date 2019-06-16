# TODO

PCEの実装。
アドホックでいい。

`tau1_series`, `tau2_series`を受け取り、終端のPCE状態を返そう。

m11 ddq1 + m12 ddq2 + h1 + g1 = t1

`q1(t, z)`を多項式カオス展開すると、

```
q1(t, z) ~= q1_0(t) phi_0(z)
            + q1_1(t) phi_1(z)
            + q1_2(t) phi_2(z)
```

選点法を使うと次の方程式が得られる。

```
[phi_0(z_1) phi_1(z_1) phi_2(z_1)] [q1_0(t)] = [q1(t, z_1)]
|phi_0(z_2) phi_1(z_2) phi_2(z_2)| |q1_1(t)|   [q1(t, z_2)]
|phi_0(z_3) phi_1(z_3) phi_2(z_3)| |q1_2(t)|   [q1(t, z_3)]
[phi_0(z_4) phi_1(z_4) phi_2(z_4)] [q1_3(t)]   [q1(t, z_4)]
```

これを解くことで多項式カオス展開の係数が得られる。
