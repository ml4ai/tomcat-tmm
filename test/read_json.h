//
// Created by Liang Zhang on 3/1/21.
//

#ifndef TOMCAT_TMM_READ_JSON_H
#define TOMCAT_TMM_READ_JSON_H

#include <iostream>

using namespace std;

class read_json {
  public:
    read_json(int i) : m_nI(i){}
    ~read_json() {}
    void testMemberFunc(int i) const { std::cout << m_nI + i << endl; }
    int  testResultFun(int i) const { return m_nI + i; }
    void testBindFun(int num) const { std::cout << m_nI + num << endl; }
    int m_nI;
};

#endif // TOMCAT_TMM_READ_JSON_H
