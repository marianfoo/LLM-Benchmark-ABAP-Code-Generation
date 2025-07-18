CLASS ltcl_test_prod_signs DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_prod_signs IMPLEMENTATION.

  METHOD test_case_1.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( 1 ) ( 2 ) ( 2 ) ( -4 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -9 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( 0 ) ( 1 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( 1 ) ( 1 ) ( 1 ) ( 2 ) ( 3 ) ( -1 ) ( 1 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -10 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    CLEAR arr.
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( 2 ) ( 4 ) ( 1 ) ( 2 ) ( -1 ) ( -1 ) ( 9 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 20 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( -1 ) ( 1 ) ( -1 ) ( 1 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( -1 ) ( 1 ) ( 1 ) ( 1 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -4 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: arr    TYPE int4_table,
          result TYPE int4.

    arr = VALUE #( ( -1 ) ( 1 ) ( 1 ) ( 0 ) ).
    result = z_humaneval_128=>prod_signs( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

ENDCLASS.