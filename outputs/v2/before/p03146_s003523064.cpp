#include <bits/stdc++.h>



using namespace std;



const int MAXN = 1000000;

int A[MAXN + 1];



int f(int n) {

	if (n % 2 == 0) {

		return n / 2;

	} else {

		return 3 * n + 1;

	}

}



int main() {

  ios::sync_with_stdio(false);

  cin.tie(0);

  int s, count = 1;

  cin >> s;



  for (int i = 0; i < MAXN + 1; i++) {

  	A[i] = 0;

  }



  A[s] += 1;

  while (true) {

  	s = f(s);

  	count++;

  	if (A[s] != 0) {

  		cout << count << '\n';

  		break;

  	}

  	A[s]++;

  }

  return 0;

}
