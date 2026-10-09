

#include <iostream>

#include <vector>

#include <map>

#include <cmath>

#include <algorithm>

using namespace std;

int main()

{

    int n,m,q;

    cin >> n >> m >> q;

    int a[q], b[q], c[q], d[q];

    for(int i=0; i<q; i++){

        cin >> a[i] >> b[i] >> c[i] >> d[i];

    }

    

    int cnt = 0;

    std::vector<int> v[100000];

    for(int i0=0; i0<m; i0++){

    for(int i1=i0; i1<m; i1++){

    for(int i2=i1; i2<m; i2++){

    for(int i3=i2; i3<m; i3++){

    for(int i4=i3; i4<m; i4++){

    for(int i5=i4; i5<m; i5++){

    for(int i6=i5; i6<m; i6++){

    for(int i7=i6; i7<m; i7++){

    for(int i8=i7; i8<m; i8++){

    for(int i9=i8; i9<m; i9++){

        v[cnt] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9}; 

        cnt++;

    }}}}}}}}}}

    

    long long int max = 0;

    for(int i=0; i<cnt; i++){

        long long int sum = 0;

        for(int j=0; j<q; j++){

            int aj = a[j]-1;

            int bj = b[j]-1;

            if(v[i][bj]-v[i][aj] == c[j]){

                sum += d[j];

            }

        }

        if(sum > max) max = sum;

    }

            

    cout << max << endl;

}