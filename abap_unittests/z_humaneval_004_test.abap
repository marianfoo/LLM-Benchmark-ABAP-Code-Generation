CLASS ltcl_test_mad DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF float_table_type,
             value TYPE f,
           END OF float_table_type,
           float_table TYPE TABLE OF float_table_type.

    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_mad IMPLEMENTATION.
  METHOD test_case_1.
    DATA: numbers  TYPE float_table,
          result   TYPE f,
          expected TYPE f,
          diff     TYPE f.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '3.0' ) ).
    expected = 2 / 3.

    result = z_humaneval_004=>mean_absolute_deviation( numbers ).
    diff = abs( result - expected ).

    cl_abap_unit_assert=>assert_true(
      act = COND #( WHEN diff < '0.000001' THEN abap_true ELSE abap_false )
      msg = 'Test case 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: numbers  TYPE float_table,
          result   TYPE f,
          expected TYPE f,
          diff     TYPE f.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '3.0' ) ( value = '4.0' ) ).
    expected = '1.0'.

    result = z_humaneval_004=>mean_absolute_deviation( numbers ).
    diff = abs( result - expected ).

    cl_abap_unit_assert=>assert_true(
      act = COND #( WHEN diff < '0.000001' THEN abap_true ELSE abap_false )
      msg = 'Test case 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: numbers  TYPE float_table,
          result   TYPE f,
          expected TYPE f,
          diff     TYPE f.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '3.0' ) ( value = '4.0' ) ( value = '5.0' ) ).
    expected = 6 / 5.

    result = z_humaneval_004=>mean_absolute_deviation( numbers ).
    diff = abs( result - expected ).

    cl_abap_unit_assert=>assert_true(
      act = COND #( WHEN diff < '0.000001' THEN abap_true ELSE abap_false )
      msg = 'Test case 3 failed'
    ).
  ENDMETHOD.
ENDCLASS.