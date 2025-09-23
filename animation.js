/* MEMO */
/*
    easing の種類
    1. linear => 一定の速度で動く
    2. ease => 始まりと終わりがゆっくり
    3. ease-in => ゆっくり始まり加速する
    4. ease-out => 速く始まり、ゆっくり終わる
    5. ease-in-out => ゆっくり始まり、加速して、またゆっくり終わる
    6. cubic-bezier( x1, y1, x2, y2 ) => カスタム曲線
*/

/************************************************************************/
// アニメーションの操作関数

// グローバル定数（#defineのような役割がある。）
const ANIMATION_TIME = 800; // アニメーションの時間（800ms）

// 汎用トランジション関数
// 引数 1. element : HTMLの要素
// 引数 2. move : アニメーション（CSS）の最終値
// 引数 3. duration : アニメーションの長さ
// 引数 4. easing : アニメーションの動き（スピード等）
// 戻り値なし
function transition ( element, move = {}, duration = ANIMATION_TIME, easing = "linear" ) {
    
    return;
}

// スクロール連携型アニメーションの定義（コールバック処理）
// 引数 1. element : HTMLの要素
// 引数 2. callback : コールバック関数
// 戻り値なし
function scrollAndAnimate ( element, callback ) {

    return;
}

// マウスカーソルを触れたときにアニメーションを切り替える関数（コールバック処理）
// 引数 1. element : HTMLの要素
// 引数 2. callbackIn : マウスカーソルが触れてるときのアニメーション等（コールバック関数）
// 引数 3. callbackOut : マウスカーソルが触れてないときのアニメーション等（コールバック関数）
// 戻り値なし
function onMouseAnimate ( element, callbackIn, callbackOut ) {

    return;
}

// クリックやタップでアニメーションを切り替える関数（コールバック処理）
// 引数 1. element : HTMLの要素
// 引数 2. callback : クリック時のアニメーション
// 戻り値なし
function tapAndAnimate ( element, callback ) {

    return;
}

/************************************************************************/
// アニメーションのプリセット関数

// フェードイン（アニメーションプリセット）
// 引数 . element : HTMLの要素
// 戻り値なし
function fadeIn ( element ) {

    return;
}

// フェードアウト（アニメーションプリセット）
// 引数 . element : HTMLの要素
// 戻り値なし
function fadeOut ( element ) {

    return;
}

// スライドアップ（アニメーションプリセット）
// 引数 . element : HTMLの要素
// 戻り値なし
function slideUp ( element ) {
    
    return;
}

// スライドダウン（アニメーションプリセット）
// 引数 . element : HTMLの要素
// 戻り値なし
function slideDown ( element ) {
    
    return;
}

// ズームイン（アニメーションプリセット）
// 引数 . element : HTMLの要素
// 戻り値なし
function zoomIn ( element ) {

    return;
}

// ズームアウト（アニメーションプリセット）
// 引数 . element : HTMLの要素
// 戻り値なし
function zoomOut ( element ) {

    return;
}

/************************************************************************/