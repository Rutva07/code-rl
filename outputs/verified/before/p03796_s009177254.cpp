#include <iostream>

#include <cmath>



using namespace std;



int main(){

  long long int n;

  cin >> n;

  long long int ans = 1;

  

  for(int i = 1; i <= n; i++){

    ans = (ans * i) % (int)(pow(10,9) + 7);

  }

  

  cout << ans << endl;

  return 0;

}