CLASS ltcl_test_change_base DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_8_base_3 FOR TESTING,
      test_9_base_3 FOR TESTING,
      test_234_base_2 FOR TESTING,
      test_16_base_2 FOR TESTING,
      test_8_base_2 FOR TESTING,
      test_7_base_2 FOR TESTING,
      test_range_2_to_7 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_change_base IMPLEMENTATION.
  METHOD test_8_base_3.
    DATA(result) = z_humaneval_044=>change_base( x = 8 base = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '22' ).
  ENDMETHOD.

  METHOD test_9_base_3.
    DATA(result) = z_humaneval_044=>change_base( x = 9 base = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '100' ).
  ENDMETHOD.

  METHOD test_234_base_2.
    DATA(result) = z_humaneval_044=>change_base( x = 234 base = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '11101010' ).
  ENDMETHOD.

  METHOD test_16_base_2.
    DATA(result) = z_humaneval_044=>change_base( x = 16 base = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '10000' ).
  ENDMETHOD.

  METHOD test_8_base_2.
    DATA(result) = z_humaneval_044=>change_base( x = 8 base = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '1000' ).
  ENDMETHOD.

  METHOD test_7_base_2.
    DATA(result) = z_humaneval_044=>change_base( x = 7 base = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '111' ).
  ENDMETHOD.

  METHOD test_range_2_to_7.
    DATA: x        TYPE i,
          expected TYPE string,
          result   TYPE string.

    DO 6 TIMES.
      x = sy-index + 1.
      expected = |{ x }|.
      result = z_humaneval_044=>change_base( x = x base = x + 1 ).
      cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
    ENDDO.
  ENDMETHOD.
ENDCLASS.