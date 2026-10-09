#include<bits/stdc++.h>

using namespace std;

int main(){

	int n,k;cin>>n>>k;int ans=0;

	for(int x=0;x<n;x++){

		int a;cin>>a;if(a>=k)ans++;

	}

	cout<<ans<<endl;

}