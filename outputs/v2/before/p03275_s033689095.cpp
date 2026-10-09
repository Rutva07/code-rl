/*

  mail: mleautomaton@foxmail.com

  author: MLEAutoMaton

  This Code is made by MLEAutoMaton

*/

#include<stdio.h>

#include<stdlib.h>

#include<string.h>

#include<math.h>

#include<algorithm>

#include<queue>

#include<set>

#include<map>

#include<iostream>

using namespace std;

#define ll long long

#define re register

#define file(a) freopen(a".in","r",stdin);freopen(a".out","w",stdout)

inline int gi(){

	int f=1,sum=0;char ch=getchar();

	while(ch>'9' || ch<'0'){if(ch=='-')f=-1;ch=getchar();}

	while(ch>='0' && ch<='9'){sum=(sum<<3)+(sum<<1)+ch-'0';ch=getchar();}

	return f*sum;

}

const int N=100010;

int a[N],b[N],n,tot,p[N],c[N<<2],Mx;

ll cnt;

int lowbit(int x){return x&(-x);}

void Add(int x,int d){while(x<=Mx){c[x]+=d;x+=lowbit(x);}}

int sum(int x){int ret=0;while(x){ret+=c[x];x-=lowbit(x);}return ret;}

bool check(int mid){

	for(int i=1;i<=n;i++)

		if(p[i]<mid)a[i]=-1;

		else a[i]=1;

	ll ans=0;

	for(int i=1;i<=n;i++){a[i]+=a[i-1];if(a[i]>=0)ans++;}

	memset(c,0,sizeof(c));Mx=0;

	for(int i=1;i<=n;i++){a[i]+=n+1;Mx=max(Mx,a[i]);}

	for(int i=1;i<=n;i++){

		ans+=sum(a[i]);

		Add(a[i],1);

	}

	return cnt-ans<cnt/2+1;

}

int q[100010];

int main(){

	n=gi();

	for(int i=1;i<=n;i++)p[i]=q[i]=gi();

	sort(q+1,q+n+1);int tot=unique(q+1,q+n+1)-q-1;

	for(int i=1;i<=n;i++)p[i]=lower_bound(q+1,q+tot+1,p[i])-q;

	int l=1,r=tot,ans=0;cnt=1ll*n*(n-1)/2ll+n;

	while(l<=r){

		int mid=(l+r)>>1;

		if(check(mid)){ans=mid;l=mid+1;}

		else r=mid-1;

	}

	printf("%d\n",q[ans]);

	return 0;

}
