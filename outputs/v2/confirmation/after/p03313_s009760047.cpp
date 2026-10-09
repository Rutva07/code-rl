#include<bits/stdc++.h>

using namespace std;

typedef long long ll;



ll read(){

    ll x=0,f=1;char c=getchar();

    while(c<'0' || c>'9'){if(c=='-')f=-1;c=getchar();}

    while(c>='0' && c<='9'){x=x*10+c-'0';c=getchar();}

    return x*f;

}



int n;

int a[1000100];

pair<int,int> b[1000100];



void upd(int x,int y){

    int num1=b[x].first,num2=b[x].second,num3=b[y].first,num4=b[y].second;

    b[x].first=max(num1,num3);

    if(num1>num3) b[x].second=max(num2,num3);

    else b[x].second=max(num1,num4);

}



int main(){

    #ifdef LZT

    freopen("in","r",stdin);

    #endif

    n=read();

    for(int i=0;i<(1<<n);i++)

        a[i]=read();

    for(int i=0;i<(1<<n);i++)

        b[i].first=a[i],b[i].second=-1e9;

    for(int k=0;k<n;k++){

        //cout<<k<<endl;

        for(int i=0;i<(1<<n);i++){

            if((i&(1<<k))!=0) continue;

            upd(i|(1<<k),i);

            //cout<<(i|(1<<k))<<' '<<i<<endl;

        }

    }/*

    for(int i=0;i+1<(1<<n);i++){

        upd(i+1,i);

    }*/

    int lastans=0;

    for(int i=1;i<(1<<n);i++){

        lastans=max(lastans,b[i].first+b[i].second);

        printf("%d\n",lastans);

    }

    return 0;

}



/*

2

1 2 3 1

*/