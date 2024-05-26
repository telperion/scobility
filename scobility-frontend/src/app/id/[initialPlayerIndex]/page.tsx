"use client";

import Home from '../../page';

export default function Page({params}: { params: {initialPlayerIndex: number}}) {
    return Home({params});
}