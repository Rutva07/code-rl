#include<bits/stdc++.h>

using namespace std;

int main()

{

    long long int n,m,i,r,s=0,b,c,j;

    scanf("%lld",&n);

    long long int a[n+10];

    long long int ar[100001]={0};

    for(i=0;i<n;i++){

        cin>>a[i];

        s+=a[i];

        ar[a[i]]++;

    }

    cin>>m;

    long long int f[m+10];

    for(i=0;i<m;i++){

        cin>>b>>c;

        s+=(c-b)*ar[b];

        ar[c]+=ar[b];

        ar[b]=0;

        cout<<s<<endl;

    }



}
