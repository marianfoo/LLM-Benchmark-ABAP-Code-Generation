CLASS ltc_test_special_factorial DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_4 FOR TESTING,
      test_5 FOR TESTING,
      test_7 FOR TESTING,
      test_1 FOR TESTING.

ENDCLASS.

CLASS ltc_test_special_factorial IMPLEMENTATION.

  METHOD test_4.
    DATA: result TYPE int8.
    result = z_humaneval_139=>special_factorial( 4 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 288
      msg = 'Test 4'
    ).
  ENDMETHOD.

  METHOD test_5.
    DATA: result TYPE int8.
    result = z_humaneval_139=>special_factorial( 5 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 34560
      msg = 'Test 5'
    ).
  ENDMETHOD.

  METHOD test_7.
    DATA: result TYPE int8.
    result = z_humaneval_139=>special_factorial( 7 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 125411328000
      msg = 'Test 7'
    ).
  ENDMETHOD.

  METHOD test_1.
    DATA: result TYPE int8.
    result = z_humaneval_139=>special_factorial( 1 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 1
      msg = 'Test 1'
    ).
  ENDMETHOD.

ENDCLASS.