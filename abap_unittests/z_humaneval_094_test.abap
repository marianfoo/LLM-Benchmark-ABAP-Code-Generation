CLASS ltc_test_z_humaneval_094 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING.

ENDCLASS.

CLASS ltc_test_z_humaneval_094 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 0 ) ( 3 ) ( 2 ) ( 1 ) ( 3 ) ( 5 ) ( 7 ) ( 4 ) ( 5 ) ( 5 ) ( 5 ) ( 2 ) ( 181 ) ( 32 ) ( 4 ) ( 32 ) ( 3 ) ( 2 ) ( 32 ) ( 324 ) ( 4 ) ( 3 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 10 msg = 'This prints if this assert fails 1 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 1 ) ( 0 ) ( 1 ) ( 8 ) ( 2 ) ( 4597 ) ( 2 ) ( 1 ) ( 3 ) ( 40 ) ( 1 ) ( 2 ) ( 1 ) ( 2 ) ( 4 ) ( 2 ) ( 5 ) ( 1 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 25 msg = 'This prints if this assert fails 2 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 1 ) ( 3 ) ( 1 ) ( 32 ) ( 5107 ) ( 34 ) ( 83278 ) ( 109 ) ( 163 ) ( 23 ) ( 2323 ) ( 32 ) ( 30 ) ( 1 ) ( 9 ) ( 3 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 13 msg = 'This prints if this assert fails 3 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 0 ) ( 724 ) ( 32 ) ( 71 ) ( 99 ) ( 32 ) ( 6 ) ( 0 ) ( 5 ) ( 91 ) ( 83 ) ( 0 ) ( 5 ) ( 6 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 11 msg = 'This prints if this assert fails 4 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 0 ) ( 81 ) ( 12 ) ( 3 ) ( 1 ) ( 21 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 msg = 'This prints if this assert fails 5 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 0 ) ( 8 ) ( 1 ) ( 2 ) ( 1 ) ( 7 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 7 msg = 'This prints if this assert fails 6 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 8191 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 19 msg = 'This prints if this assert fails 7 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 8191 ) ( 123456 ) ( 127 ) ( 7 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 19 msg = 'This prints if this assert fails 8 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: lst    TYPE int4_table,
          result TYPE i.

    CLEAR lst.
    lst = VALUE #( ( 127 ) ( 97 ) ( 8192 ) ).
    result = z_humaneval_094=>skjkasdkd( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 10 msg = 'This prints if this assert fails 9 (also good for debugging!)' ).
  ENDMETHOD.

ENDCLASS.