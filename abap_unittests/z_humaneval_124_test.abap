CLASS ltcl_test_valid_date DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_valid_date_03_11_2000 FOR TESTING,
      test_invalid_date_15_01_2012 FOR TESTING,
      test_invalid_date_04_0_2040 FOR TESTING,
      test_valid_date_06_04_2020 FOR TESTING,
      test_valid_date_01_01_2007 FOR TESTING,
      test_invalid_date_03_32_2011 FOR TESTING,
      test_invalid_empty_date FOR TESTING,
      test_invalid_date_04_31_3000 FOR TESTING,
      test_valid_date_06_06_2005 FOR TESTING,
      test_invalid_date_21_31_2000 FOR TESTING,
      test_valid_date_04_12_2003 FOR TESTING,
      test_invalid_date_04122003 FOR TESTING,
      test_invalid_date_20030412 FOR TESTING,
      test_invalid_date_2003_04 FOR TESTING,
      test_invalid_date_2003_04_12 FOR TESTING,
      test_invalid_date_04_2003 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_valid_date IMPLEMENTATION.
  METHOD test_valid_date_03_11_2000.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '03-11-2000' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_invalid_date_15_01_2012.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '15-01-2012' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_date_04_0_2040.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '04-0-2040' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_valid_date_06_04_2020.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '06-04-2020' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_valid_date_01_01_2007.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '01-01-2007' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_invalid_date_03_32_2011.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '03-32-2011' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_empty_date.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_date_04_31_3000.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '04-31-3000' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_valid_date_06_06_2005.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '06-06-2005' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_invalid_date_21_31_2000.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '21-31-2000' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_valid_date_04_12_2003.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '04-12-2003' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_invalid_date_04122003.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '04122003' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_date_20030412.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '20030412' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_date_2003_04.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '2003-04' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_date_2003_04_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '2003-04-12' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_invalid_date_04_2003.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_124=>valid_date( '04-2003' )
      exp = abap_false ).
  ENDMETHOD.
ENDCLASS.