CLASS ltcl_test_search DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING,
      test_case_12 FOR TESTING,
      test_case_13 FOR TESTING,
      test_case_14 FOR TESTING,
      test_case_15 FOR TESTING,
      test_case_16 FOR TESTING,
      test_case_17 FOR TESTING,
      test_case_18 FOR TESTING,
      test_case_19 FOR TESTING,
      test_case_20 FOR TESTING,
      test_case_21 FOR TESTING,
      test_case_22 FOR TESTING,
      test_case_23 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_search IMPLEMENTATION.
  METHOD test_case_1.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 5 ) ( 5 ) ( 5 ) ( 5 ) ( 1 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 4 ) ( 1 ) ( 4 ) ( 1 ) ( 4 ) ( 4 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 3 ) ( 3 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 8 ) ( 8 ) ( 8 ) ( 8 ) ( 8 ) ( 8 ) ( 8 ) ( 8 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 8 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 2 ) ( 3 ) ( 3 ) ( 2 ) ( 2 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 2 ) ( 7 ) ( 8 ) ( 8 ) ( 4 ) ( 8 ) ( 7 ) ( 3 ) ( 9 ) ( 6 ) ( 5 ) ( 10 ) ( 4 ) ( 3 ) ( 6 ) ( 7 ) ( 1 ) ( 7 ) ( 4 ) ( 10 ) ( 8 ) ( 1 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 3 ) ( 2 ) ( 8 ) ( 2 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 6 ) ( 7 ) ( 1 ) ( 8 ) ( 8 ) ( 10 ) ( 5 ) ( 8 ) ( 5 ) ( 3 ) ( 10 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 8 ) ( 8 ) ( 3 ) ( 6 ) ( 5 ) ( 6 ) ( 4 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 6 ) ( 9 ) ( 6 ) ( 7 ) ( 1 ) ( 4 ) ( 7 ) ( 1 ) ( 8 ) ( 8 ) ( 9 ) ( 8 ) ( 10 ) ( 10 ) ( 8 ) ( 4 ) ( 10 ) ( 4 ) ( 10 ) ( 1 ) ( 2 ) ( 9 ) ( 5 ) ( 7 ) ( 9 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 1 ) ( 9 ) ( 10 ) ( 1 ) ( 3 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 6 ) ( 9 ) ( 7 ) ( 5 ) ( 8 ) ( 7 ) ( 5 ) ( 3 ) ( 7 ) ( 5 ) ( 10 ) ( 10 ) ( 3 ) ( 6 ) ( 10 ) ( 2 ) ( 8 ) ( 6 ) ( 5 ) ( 4 ) ( 9 ) ( 5 ) ( 3 ) ( 10 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_case_13.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 1 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_14.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 8 ) ( 8 ) ( 10 ) ( 6 ) ( 4 ) ( 3 ) ( 5 ) ( 8 ) ( 2 ) ( 4 ) ( 2 ) ( 8 ) ( 4 ) ( 6 ) ( 10 ) ( 4 ) ( 2 ) ( 1 ) ( 10 ) ( 2 ) ( 1 ) ( 1 ) ( 5 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_15.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 2 ) ( 10 ) ( 4 ) ( 8 ) ( 2 ) ( 10 ) ( 5 ) ( 1 ) ( 2 ) ( 9 ) ( 5 ) ( 5 ) ( 6 ) ( 3 ) ( 8 ) ( 6 ) ( 4 ) ( 10 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_16.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 1 ) ( 6 ) ( 10 ) ( 1 ) ( 6 ) ( 9 ) ( 10 ) ( 8 ) ( 6 ) ( 8 ) ( 7 ) ( 3 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_17.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 9 ) ( 2 ) ( 4 ) ( 1 ) ( 5 ) ( 1 ) ( 5 ) ( 2 ) ( 5 ) ( 7 ) ( 7 ) ( 7 ) ( 3 ) ( 10 ) ( 1 ) ( 5 ) ( 4 ) ( 2 ) ( 8 ) ( 4 ) ( 1 ) ( 9 ) ( 10 ) ( 7 ) ( 10 ) ( 2 ) ( 8 ) ( 10 ) ( 9 ) ( 4 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_18.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 2 ) ( 6 ) ( 4 ) ( 2 ) ( 8 ) ( 7 ) ( 5 ) ( 6 ) ( 4 ) ( 10 ) ( 4 ) ( 6 ) ( 3 ) ( 7 ) ( 8 ) ( 8 ) ( 3 ) ( 1 ) ( 4 ) ( 2 ) ( 2 ) ( 10 ) ( 7 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_19.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 9 ) ( 8 ) ( 6 ) ( 10 ) ( 2 ) ( 6 ) ( 10 ) ( 2 ) ( 7 ) ( 8 ) ( 10 ) ( 3 ) ( 8 ) ( 2 ) ( 6 ) ( 2 ) ( 3 ) ( 1 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_20.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 5 ) ( 5 ) ( 3 ) ( 9 ) ( 5 ) ( 6 ) ( 3 ) ( 2 ) ( 8 ) ( 5 ) ( 6 ) ( 10 ) ( 10 ) ( 6 ) ( 8 ) ( 4 ) ( 10 ) ( 7 ) ( 7 ) ( 10 ) ( 8 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_21.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 10 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_22.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 9 ) ( 7 ) ( 7 ) ( 2 ) ( 4 ) ( 7 ) ( 2 ) ( 10 ) ( 9 ) ( 7 ) ( 5 ) ( 7 ) ( 2 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_23.
    DATA: lst    TYPE int4_table,
          result TYPE i.
    lst = VALUE #( ( 3 ) ( 10 ) ( 10 ) ( 9 ) ( 2 ) ).
    result = z_humaneval_069=>search( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.
ENDCLASS.