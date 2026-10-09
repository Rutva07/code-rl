#include<cstdio>

#include<cstring>

#include<cstdlib>

#include<cmath>

#include<algorithm>

#include<iostream>

using namespace std;

int main(){

	int A,B,C;

	cin>>A>>B>>C;

	int n;

	n=C%B;

	for(int i=1;i<=1000;i++){

		if((n+B*i)%A==0){

			printf("YES\n");

			return 0;

		}

	}

	printf("NO\n");

}
