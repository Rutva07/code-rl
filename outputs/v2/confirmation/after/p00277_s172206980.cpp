#include<iostream>

#include<algorithm>

#include<map>

using namespace std;

typedef pair<int,int> P;

#define fr first

#define sc second

P seg[1<<18];

int n;

void update(int id,int pt){

  int k = id + n - 1;

  seg[k].fr += pt;

  seg[k].sc = -id;

  while(k != 0){

    k = ( k - 1 ) / 2;

    seg[k] = max(seg[2*k+1],seg[2*k+2]);

  }

}

void init(int size) {

  n = 1;

  while(n < size) n *= 2;

}

int main(){

  int N,R,L;

  cin >> N >> R >> L;

  init(N);

  for(int i = 0 ; i < N ; i++ ) update(i,0);

  int mostid = 0 , now = 0 , sum[100000] = {};

  for(int i = 0 ; i < R ; i++ ){

    int d,t,x;

    cin >> d >> t >> x;

    update(d-1,x);

    sum[mostid] += t - now;

    now = t;

    mostid = -seg[0].sc;

  }

  sum[mostid] += L - now;

  int rec = 0;

  for(int i = 0 ; i < N ; i++) if( sum[rec] < sum[i] ) rec = i;

  cout << rec + 1 << endl;

}