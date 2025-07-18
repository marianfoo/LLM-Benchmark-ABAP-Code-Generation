CLASS ltcl_test_intersection DEFINITION
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

CLASS ltcl_test_intersection IMPLEMENTATION.

  METHOD test_case_1.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( 1 ) ( 2 ) ).
    interval2 = VALUE #( ( 2 ) ( 3 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( -1 ) ( 1 ) ).
    interval2 = VALUE #( ( 0 ) ( 4 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( -3 ) ( -1 ) ).
    interval2 = VALUE #( ( -5 ) ( 5 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'YES' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( -2 ) ( 2 ) ).
    interval2 = VALUE #( ( -4 ) ( 0 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'YES' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( -11 ) ( 2 ) ).
    interval2 = VALUE #( ( -1 ) ( -1 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( 1 ) ( 2 ) ).
    interval2 = VALUE #( ( 3 ) ( 5 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( 1 ) ( 2 ) ).
    interval2 = VALUE #( ( 1 ) ( 2 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: interval1 TYPE int4_table,
          interval2 TYPE int4_table,
          result    TYPE string.

    interval1 = VALUE #( ( -2 ) ( -2 ) ).
    interval2 = VALUE #( ( -3 ) ( -2 ) ).
    result = z_humaneval_127=>intersection( interval1 = interval1 interval2 = interval2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

ENDCLASS.