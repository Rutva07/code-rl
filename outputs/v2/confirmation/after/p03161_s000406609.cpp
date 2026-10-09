#include<bits/stdc++.h>

using namespace std;

#define ll long long

#define vi vector<int>

#define pb push_back

#define pii pair<int , int>

#define fast ios_base::sync_with_stdio(0)

#define mx 100005

#define inf 100000000000000000LL

ll arr[mx], dp[mx];

int n, k;

ll solve(int pos)

{

    if(pos==n) return 0;

    ll&ret = dp[pos];

    if(dp[pos]!=-1) return ret;

    ret = inf;

    for(int i=1; i<=k; i++){

        int nxt = pos+i;

        if(nxt>n) break;

        ret = min(ret, llabs(arr[pos] - arr[nxt]) + solve(nxt));

    }

    return ret;

}

int main()

{

    //fast;

    while(scanf("%d%d",&n,&k)){

        for(int i=1; i<=n; i++)

            scanf("%lld",&arr[i]);

        memset(dp, -1, sizeof dp);

        printf("%lld\n", solve(1));

        return 0;

    }

}


